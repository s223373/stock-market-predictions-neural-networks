"""
scheduler.py
============
Runs every 5 minutes, updates the dataframe, checks feature signals,
and runs the LSTM model. Alerts are printed to the terminal and shown
as a macOS notification.

Usage
-----
    python scheduler.py --model StockPriceLSTMNetwork_2026-05-29_17-06-41.pt
"""

import time
import datetime
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import schedule
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from features import build_features, FEATURES
from lstm_backtest import StockPriceLSTMNetwork

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TICKER      = "AAPL"
WINDOW_SIZE = 14
PRED_STEPS  = 14
THRESHOLD   = 1.0    # % move to trigger BUY / SELL

ALERT_FEATURES = {
    "macd_bullish_entry": "MACD Bullish Entry",
    "macd_bearish_entry": "MACD Bearish Entry",
    "rsi_overbought":     "RSI Overbought (>70)",
    "rsi_oversold":       "RSI Oversold (<30)",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def notify(title: str, message: str):
    """macOS notification banner."""
    subprocess.run([
        "osascript", "-e",
        f'display notification "{message}" with title "{title}" sound name "Default"'
    ])

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str) -> StockPriceLSTMNetwork:
    model = StockPriceLSTMNetwork(
        input_size  = len(FEATURES),
        hidden_size = 64,
        output_size = 1,
        num_layers  = 1,
        dropout     = 0.3,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info(f"Model loaded  ←  {path}")
    return model


def run_lstm(model: StockPriceLSTMNetwork, df: pd.DataFrame) -> tuple[str, float]:
    raw = df[FEATURES].dropna().values.astype(float)
    if len(raw) < WINDOW_SIZE + PRED_STEPS:
        log.warning("Not enough data for prediction.")
        return "HOLD", 0.0

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(raw)
    norm = scaler.transform(raw)

    close_scaler = MinMaxScaler(feature_range=(-1, 1))
    close_scaler.fit(raw[:, 0].reshape(-1, 1))

    input_seq = torch.FloatTensor(norm[-WINDOW_SIZE:])

    preds_norm = []
    with torch.no_grad():
        for step in range(PRED_STEPS):
            seq  = input_seq if step == 0 else torch.FloatTensor(last_window)
            pred = model(seq).item()
            preds_norm.append(pred)
            new_row    = seq[-1].clone()
            new_row[0] = pred
            last_window = torch.cat([seq[1:], new_row.unsqueeze(0)], dim=0).numpy()
            input_seq   = torch.FloatTensor(last_window)

    preds_inv  = close_scaler.inverse_transform(
        np.array(preds_norm).reshape(-1, 1)
    ).flatten()

    pct_change = (preds_inv[-1] - preds_inv[0]) / (preds_inv[0] + 1e-9) * 100

    if pct_change > THRESHOLD:   return "BUY",  pct_change
    if pct_change < -THRESHOLD:  return "SELL", pct_change
    return "HOLD", pct_change

# ─────────────────────────────────────────────────────────────────────────────
# MAIN JOB
# ─────────────────────────────────────────────────────────────────────────────

def run_job(model: StockPriceLSTMNetwork):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    log.info(f"── Checking {TICKER} at {now} ──")

    # 1. Fresh data
    try:
        df = yf.download(TICKER, period="5d", interval="5m", progress=False)
        df.columns = df.columns.get_level_values(0)
        df.index   = pd.to_datetime(df.index)
    except Exception as e:
        log.error(f"Download failed: {e}")
        return

    # 2. Features
    df     = build_features(df)
    latest = df.iloc[-1]
    price  = float(latest["Close"])

    # 3. Check alert features
    triggered = [
        label
        for col, label in ALERT_FEATURES.items()
        if col in df.columns and bool(latest.get(col, False))
    ]

    # 4. Run LSTM
    signal, pct_change = run_lstm(model, df)

    # 5. Print status
    log.info(f"Price: ${price:.2f}  |  LSTM: {signal} ({pct_change:+.2f}%)")
    if triggered:
        log.info("Signals: " + ", ".join(triggered))
    else:
        log.info("Signals: none")

    # 6. macOS notification if anything fired
    if triggered or signal in ("BUY", "SELL"):
        lines = []
        if triggered:
            lines.append(", ".join(triggered))
        if signal in ("BUY", "SELL"):
            lines.append(f"LSTM: {signal} ({pct_change:+.2f}%)")
        notify(
            title   = f"{TICKER}  ${price:.2f}",
            message = "  |  ".join(lines),
        )

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--interval", type=int, default=5, help="Minutes between checks (default: 5)")
    args = parser.parse_args()

    model = load_model(args.model)

    run_job(model)
    schedule.every(args.interval).minutes.do(run_job, model=model)

    log.info(f"Scheduler running every {args.interval} min — press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)
