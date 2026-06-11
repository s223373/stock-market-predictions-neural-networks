"""
alpaca_trader.py
================
Live paper/live trading bot wired to StockPriceLSTMNetworkDualStream.

Requires
--------
    pip install alpaca-py torch joblib

Place this file next to model.py, feature_engineering.py, and your saved
    StockPriceLSTMNetwork_<timestamp>.pt   ← produced by train.py

The checkpoint bundles everything (weights, scaler, bool_cols, window_size)
so no separate close_scaler.pkl is needed.

Usage
-----
    python alpaca_trader.py
"""

import time
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from feature_engineering import build_features, FEATURES
from model import StockPriceLSTMNetworkDualStream, prepare_price_input

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit everything in this block; touch nothing else
# ─────────────────────────────────────────────────────────────────────────────

API_KEY    = "YOUR_ALPACA_API_KEY"
SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
PAPER      = True          # True = paper trading; False = live money

SYMBOL      = "SPY"        # ticker to trade — match TICKER in train.py
FETCH_EXTRA = 250          # extra bars fetched so indicators can warm up

# Trading thresholds (model output is a predicted scaled Close value)
BUY_THRESHOLD  =  0.001   # pred - last_close > this  → BUY
SELL_THRESHOLD = -0.001   # pred - last_close < this  → SELL
ORDER_QTY      = 1        # shares per order (keep small during testing)

# Path to the checkpoint produced by train.py
MODEL_PATH = "StockPriceLSTMNetwork_YYYY-MM-DD_HH-MM-SS.pt"  # ← update this

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
# LOAD CHECKPOINT  (weights + scaler + config all in one file)
# ─────────────────────────────────────────────────────────────────────────────

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Pull everything the trainer saved
N_BOOL_FEATURES = checkpoint["n_bool_features"]
HIDDEN_SIZE     = checkpoint["hidden_size"]
BOOL_COLS       = checkpoint["bool_cols"]
close_scaler    = checkpoint["close_scaler"]
LOOKBACK        = checkpoint["window_size"]

# Sanity-check: checkpoint bool_cols must match current FEATURES list
expected_bool = [f for f in FEATURES if f != "Close"]
assert BOOL_COLS == expected_bool, (
    "Checkpoint bool_cols don't match current FEATURES.\n"
    f"  checkpoint : {BOOL_COLS}\n"
    f"  FEATURES   : {expected_bool}\n"
    "Re-train or update feature_engineering.py to match."
)

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE CLIENTS + MODEL
# ─────────────────────────────────────────────────────────────────────────────

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client    = StockHistoricalDataClient(API_KEY, SECRET_KEY)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
log.info("Using device: %s", device)

model = StockPriceLSTMNetworkDualStream(
    n_bool_features=N_BOOL_FEATURES,
    hidden_size=HIDDEN_SIZE,
    output_size=1,
    num_layers=1,
    dropout=0.3,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
log.info(
    "Loaded checkpoint %s  |  window=%d  n_bool=%d  hidden=%d",
    MODEL_PATH, LOOKBACK, N_BOOL_FEATURES, HIDDEN_SIZE,
)

# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS CHECK
# ─────────────────────────────────────────────────────────────────────────────

def market_is_open() -> bool:
    """Return True only when the US equity market is currently open."""
    clock = trading_client.get_clock()
    return clock.is_open

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_bars() -> pd.DataFrame:
    """
    Pull the most recent 5-minute bars from Alpaca and return a raw OHLCV
    DataFrame with a tz-aware DatetimeIndex (America/New_York), which is
    what build_features() expects.
    """
    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        limit=LOOKBACK + FETCH_EXTRA,
    )
    bars = data_client.get_stock_bars(request).df

    # Alpaca returns a MultiIndex (symbol, timestamp); drop the symbol level
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index(level=0, drop=True)

    # Normalise column names to match yfinance convention used in training
    bars.columns = [c.capitalize() for c in bars.columns]
    # Alpaca uses 'Vwap'; drop it — build_features() doesn't need it
    bars = bars[["Open", "High", "Low", "Close", "Volume"]]

    # Alpaca timestamps are UTC; convert to NY time to match training data
    bars.index = bars.index.tz_convert("America/New_York")

    return bars

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def get_prediction() -> float:
    """
    Fetch live bars → feature-engineer → run the dual-stream model → return
    the raw scalar prediction (predicted next-bar return).
    """
    bars = fetch_bars()

    # build_features downloads SPY/QQQ internally; pass matching period/interval
    features_df = build_features(bars, period="5d", interval="5m")

    # Keep only the columns the model was trained on, in the correct order
    features_df = features_df[FEATURES].dropna()

    if len(features_df) < LOOKBACK:
        raise RuntimeError(
            f"Not enough clean rows after feature engineering: "
            f"{len(features_df)} < {LOOKBACK}. "
            "Increase FETCH_EXTRA or check for NaN-heavy columns."
        )

    window = features_df.iloc[-LOOKBACK:]   # (LOOKBACK, n_features)

    # ── Close → scaled → z-scored first differences (Stream 1) ───────────────
    close_raw    = window["Close"].values.reshape(-1, 1)
    close_scaled = close_scaler.transform(close_raw).flatten()          # (LOOKBACK,)
    close_tensor = torch.tensor(close_scaled, dtype=torch.float32)
    price_input  = prepare_price_input(close_tensor)              # (LOOKBACK-1, 1)
    price_input  = price_input.unsqueeze(0).to(device)            # (1, LOOKBACK-1, 1)

    # ── Boolean columns (Stream 2) ────────────────────────────────────────────
    bool_vals   = window[BOOL_COLS].values.astype(np.float32)     # (LOOKBACK, n_bool)
    bool_tensor = torch.tensor(bool_vals, dtype=torch.float32)
    bool_input  = bool_tensor[1:, :].unsqueeze(0).to(device)      # (1, LOOKBACK-1, n_bool)
    # Drop first row to align with the (LOOKBACK-1) price difference sequence

    with torch.no_grad():
        pred = model(price_input, bool_input).item()

    # pred is a scaled Close value; compute delta vs the last known bar
    last_close_scaled = float(close_scaled[-1])
    return pred, last_close_scaled

# ─────────────────────────────────────────────────────────────────────────────
# POSITION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_position_qty(symbol: str) -> int:
    """Return current held shares (0 if no position)."""
    try:
        pos = trading_client.get_open_position(symbol)
        return int(float(pos.qty))
    except Exception:
        return 0

# ─────────────────────────────────────────────────────────────────────────────
# ORDER EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def execute_signal(pred: float, last_close_scaled: float) -> None:
    """
    Map model output to a trade decision and submit to Alpaca.

    The model predicts the next bar's scaled Close value. We compute the
    delta vs the last known scaled Close and threshold on that.

    Rules
    -----
    delta > BUY_THRESHOLD  and no position  → BUY
    delta < SELL_THRESHOLD and long position → SELL (close)
    otherwise                                → HOLD
    """
    delta       = pred - last_close_scaled
    current_qty = get_position_qty(SYMBOL)

    if delta > BUY_THRESHOLD and current_qty == 0:
        side = OrderSide.BUY
    elif delta < SELL_THRESHOLD and current_qty > 0:
        side = OrderSide.SELL
    else:
        log.info("HOLD  delta=%.5f  pred=%.5f  last=%.5f  position=%d",
                 delta, pred, last_close_scaled, current_qty)
        return

    order = MarketOrderRequest(
        symbol=SYMBOL,
        qty=ORDER_QTY,
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    result = trading_client.submit_order(order)
    log.info(
        "ORDER  side=%-4s  symbol=%s  qty=%s  id=%s  delta=%.5f",
        side.value, SYMBOL, ORDER_QTY, result.id, delta,
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("Bot started  symbol=%s  paper=%s  device=%s", SYMBOL, PAPER, device)
    while True:
        try:
            if not market_is_open():
                log.info("Market closed — sleeping 60 s")
                time.sleep(60)
                continue

            pred, last_close_scaled = get_prediction()
            log.info("Prediction: %.5f  last_scaled: %.5f  delta: %.5f",
                     pred, last_close_scaled, pred - last_close_scaled)
            execute_signal(pred, last_close_scaled)

        except Exception as exc:
            log.error("Error in main loop: %s", exc, exc_info=True)

        # Sleep until the next 5-minute bar. A small offset (10 s) lets the
        # bar fully close and appear in Alpaca's feed before we fetch.
        time.sleep(5 * 60 + 10)


if __name__ == "__main__":
    main()