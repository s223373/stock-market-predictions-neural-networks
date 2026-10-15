"""
nb_bunched_trader.py
====================
Live paper/live trading bot for the bunched Naive Bayes models.

How it works
------------
Every time a 1m bar closes, the last WINDOW (5) bars form one bunched sample.
The sample's boolean features = max over those 5 bars (same roll-up as
training). Model number  (epoch_minute % WINDOW)  scores it - so the 5
offset-models take turns, one per minute.

BUY / SELL prediction while flat -> bracket order (target + stop on Alpaca's
side). A max-hold timeout and an end-of-day flatten are handled here.
Entries are always "end" mode: we trade after the 5-bar window has closed.

Requires
--------
    pip install alpaca-py scikit-learn joblib

Set your keys as environment variables (never hardcode them):
    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...

Place next to feature_engineering.py and the NBBunched_<timestamp>.joblib
produced by train_nb_bunched.py.

Usage
-----
    python nb_bunched_trader.py
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest, MarketOrderRequest, StopLossRequest, TakeProfitRequest,
)

from feature_engineering import build_features

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit everything in this block; touch nothing else
# ─────────────────────────────────────────────────────────────────────────────

API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
PAPER      = True            # True = paper trading; False = live money

SYMBOL      = "SPY"          # match the ticker the models were trained on
DATA_FEED   = "iex"          # "iex" (free plan) or "sip" (paid plan)
FETCH_BARS  = 600            # 1m bars fetched each cycle so indicators can warm up
ORDER_QTY   = 1              # whole shares (bracket orders need whole shares)
ALLOW_SHORT = False          # True needs a margin account + a shortable symbol

MODEL_PATH = "NBBunched_2026-10-15_00-00-00.joblib"   # ← update this

# Overrides for the exit settings stored in the checkpoint (None = use checkpoint)
TARGET_PCT_OVERRIDE    = None
STOP_PCT_OVERRIDE      = None
MAX_HOLD_MIN_OVERRIDE  = None

FLATTEN_BEFORE_CLOSE_MIN = 5     # close everything this many minutes before the bell
BAR_SETTLE_SECONDS       = 5     # wait after the minute rolls so the bar is published
MAX_STALENESS_SECONDS    = 150   # skip if the newest closed bar is older than this
STATE_PATH = "nb_trader_state.json"

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

checkpoint = joblib.load(MODEL_PATH)

MODELS       = checkpoint["models"]
BOOL_COLS    = checkpoint["bool_cols"]
WINDOW       = checkpoint["window"]
TARGET_PCT   = TARGET_PCT_OVERRIDE   if TARGET_PCT_OVERRIDE   is not None else checkpoint["target_pct"]
STOP_PCT     = STOP_PCT_OVERRIDE     if STOP_PCT_OVERRIDE     is not None else checkpoint["stop_pct"]
MAX_HOLD_MIN = MAX_HOLD_MIN_OVERRIDE if MAX_HOLD_MIN_OVERRIDE is not None else checkpoint["max_hold_bars"]

BUY, HOLD, SELL = 2, 1, 0

if checkpoint.get("ticker") != SYMBOL:
    log.warning("Models were trained on %s but SYMBOL=%s - make sure this is intended.",
                checkpoint.get("ticker"), SYMBOL)
if checkpoint.get("interval") != "1m":
    log.warning("Checkpoint interval is %s, trader assumes 1m bars.", checkpoint.get("interval"))

# ─────────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────────

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client    = StockHistoricalDataClient(API_KEY, SECRET_KEY)
FEED = DataFeed.IEX if DATA_FEED.lower() == "iex" else DataFeed.SIP

log.info("Loaded %s | models=%d window=%d bool_features=%d | target=%.3f%% stop=%.3f%% hold=%dm",
         MODEL_PATH, len(MODELS), WINDOW, len(BOOL_COLS),
         TARGET_PCT * 100, STOP_PCT * 100, MAX_HOLD_MIN)

# ─────────────────────────────────────────────────────────────────────────────
# STATE  (entry time survives restarts so the timeout still works)
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)

# ─────────────────────────────────────────────────────────────────────────────
# MARKET / POSITION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_position_qty(symbol: str) -> int:
    """Signed position size: >0 long, <0 short, 0 flat."""
    try:
        pos = trading_client.get_open_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0


def open_orders(symbol: str):
    return trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )


def flatten(reason: str) -> None:
    """Cancel resting bracket legs, then close the position at market."""
    for o in open_orders(SYMBOL):
        try:
            trading_client.cancel_order_by_id(o.id)
        except Exception as exc:
            log.warning("Could not cancel order %s: %s", o.id, exc)
    try:
        trading_client.close_position(SYMBOL)
        log.info("FLATTEN  reason=%s", reason)
    except Exception as exc:
        log.warning("close_position failed (%s): %s", reason, exc)
    save_state({})

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_bars() -> pd.DataFrame:
    """Recent CLOSED 1m bars, tz-aware America/New_York, OHLCV only."""
    now = datetime.now(timezone.utc)
    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Minute,
        start=now - timedelta(days=5),     # 5 calendar days covers weekends/holidays
        feed=FEED,
    )
    bars = data_client.get_stock_bars(request).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index(level=0, drop=True)

    bars.columns = [c.capitalize() for c in bars.columns]
    bars = bars[["Open", "High", "Low", "Close", "Volume"]]

    # Alpaca bar timestamps = bar START. Keep only fully closed bars.
    bars = bars[bars.index + pd.Timedelta(minutes=1) <= now]
    bars = bars.tail(FETCH_BARS)

    bars.index = bars.index.tz_convert("America/New_York")
    return bars

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def get_prediction():
    """
    Returns (signal, bar_ts, last_close, model_offset, proba_dict) or None if
    the data isn't good enough to score this minute.
    """
    bars = fetch_bars()
    if len(bars) < WINDOW + 50:
        log.warning("Only %d bars returned - not enough to warm indicators.", len(bars))
        return None

    last_ts = bars.index[-1]
    age = (datetime.now(timezone.utc) - last_ts.tz_convert("UTC").to_pydatetime()
           - timedelta(minutes=1)).total_seconds()
    if age > MAX_STALENESS_SECONDS:
        log.warning("Newest closed bar is %.0fs old - skipping (thin/late feed?).", age)
        return None

    features_df = build_features(bars, period="5d", interval="1m", ticker=SYMBOL)

    missing = [c for c in BOOL_COLS if c not in features_df.columns]
    if missing:
        raise RuntimeError(f"Feature columns missing vs checkpoint: {missing}")

    feats = features_df[BOOL_COLS].ffill().fillna(0)      # same cleaning as training
    window = feats.iloc[-WINDOW:]

    # Gap check (same rule as bunched_datasets.DROP_GAPPED)
    span = window.index[-1] - window.index[0]
    if span > pd.Timedelta(minutes=WINDOW):
        log.warning("Last %d bars span %s (gap in data) - skipping.", WINDOW, span)
        return None

    x = pd.DataFrame([window.max(axis=0).astype(int).values], columns=BOOL_COLS)

    offset = int(last_ts.timestamp() // 60) % WINDOW      # which model's turn it is
    model = MODELS[offset]
    pred = int(model.predict(x)[0])
    proba = dict(zip(model.classes_.tolist(), np.round(model.predict_proba(x)[0], 3).tolist()))

    return pred, last_ts, float(bars["Close"].iloc[-1]), offset, proba

# ─────────────────────────────────────────────────────────────────────────────
# ORDER EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def enter(signal: int, last_close: float, state: dict) -> dict:
    if signal == BUY:
        side = OrderSide.BUY
        tp, sl = last_close * (1 + TARGET_PCT), last_close * (1 - STOP_PCT)
    else:
        side = OrderSide.SELL
        tp, sl = last_close * (1 - TARGET_PCT), last_close * (1 + STOP_PCT)

    order = MarketOrderRequest(
        symbol=SYMBOL,
        qty=ORDER_QTY,
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(tp, 2)),
        stop_loss=StopLossRequest(stop_price=round(sl, 2)),
    )
    result = trading_client.submit_order(order)
    log.info("ORDER  side=%-4s qty=%s ref=%.2f  target=%.2f stop=%.2f  id=%s",
             side.value, ORDER_QTY, last_close, tp, sl, result.id)

    state = {"entry_time": datetime.now(timezone.utc).isoformat(), "side": side.value}
    save_state(state)
    return state


def manage_open_position(state: dict, minutes_to_close: float) -> bool:
    """Handle timeout / end-of-day for an existing position. Returns True if in-trade."""
    qty = get_position_qty(SYMBOL)
    if qty == 0:
        # Flat (bracket leg hit, or entry not filled yet). Clear stale state only
        # if nothing is resting on the book either.
        if state and not list(open_orders(SYMBOL)):
            save_state({})
            return False
        return bool(state)

    if minutes_to_close <= FLATTEN_BEFORE_CLOSE_MIN:
        flatten("end of day")
        return False

    entry = state.get("entry_time")
    if entry:
        held = (datetime.now(timezone.utc) - datetime.fromisoformat(entry)).total_seconds() / 60
        if held >= MAX_HOLD_MIN:
            flatten(f"timeout after {held:.0f} min")
            return False
    else:
        log.warning("Position open but no entry time on file - timeout not tracked.")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Bot started  symbol=%s  paper=%s  feed=%s", SYMBOL, PAPER, DATA_FEED)
    state = load_state()
    last_scored = None

    while True:
        try:
            clock = trading_client.get_clock()
            if not clock.is_open:
                log.info("Market closed - sleeping 60 s")
                time.sleep(60)
                continue

            minutes_to_close = (clock.next_close - clock.timestamp).total_seconds() / 60

            # 1) Manage any existing trade first
            in_trade = manage_open_position(state, minutes_to_close)
            state = load_state()

            # 2) Score the newest completed window
            out = get_prediction()
            if out is not None:
                signal, bar_ts, last_close, offset, proba = out
                names = {BUY: "BUY", HOLD: "HOLD", SELL: "SELL"}

                if bar_ts == last_scored:
                    log.info("Bar %s already scored - waiting.", bar_ts)
                else:
                    last_scored = bar_ts
                    log.info("model=%d  bar=%s  close=%.2f  signal=%s  proba=%s  in_trade=%s",
                             offset, bar_ts.strftime("%H:%M"), last_close,
                             names[signal], proba, in_trade)

                    can_enter = (not in_trade
                                 and minutes_to_close > FLATTEN_BEFORE_CLOSE_MIN + 1)
                    if signal == BUY and can_enter:
                        state = enter(BUY, last_close, state)
                    elif signal == SELL and can_enter:
                        if ALLOW_SHORT:
                            state = enter(SELL, last_close, state)
                        else:
                            log.info("SELL signal ignored (ALLOW_SHORT=False).")

        except Exception as exc:
            log.error("Error in main loop: %s", exc, exc_info=True)

        # Sleep to just after the next minute boundary so the new bar has published
        time.sleep(60 - (time.time() % 60) + BAR_SETTLE_SECONDS)


if __name__ == "__main__":
    main()