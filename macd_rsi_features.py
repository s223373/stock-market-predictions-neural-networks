import numpy as np
import pandas as pd


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _rolling_linreg_endpoint(series, length):
    """Matches ta.linreg(close, length, 0) — endpoint of the fitted line."""
    y      = series.values.astype(float)
    result = np.full(len(y), np.nan)
    x      = np.arange(length, dtype=float)
    xm     = x.mean()

    for i in range(length - 1, len(y)):
        yw     = y[i - length + 1 : i + 1]
        ym     = yw.mean()
        slope  = np.dot(x - xm, yw - ym) / (np.dot(x - xm, x - xm) + 1e-12)
        result[i] = ym + slope * (length - 1 - xm)

    return pd.Series(result, index=series.index)


def add_macd_lr_features(df, fast=12, slow=26, signal_len=9, lr_length=100, lr_mult=4.0):
    df    = df.copy()
    close = df["Close"].squeeze()

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd   = _ema(close, fast) - _ema(close, slow)
    signal = _ema(macd, signal_len)
    df["macd_line"]   = macd
    df["macd_signal"] = signal
    df["macd_hist"]   = macd - signal

    # ── Linear Regression Channel ─────────────────────────────────────────────
    lr  = _rolling_linreg_endpoint(close, lr_length)
    dev = lr_mult * (close - lr).rolling(lr_length).std()

    df["lr_upper"] = lr + dev
    df["lr_lower"] = lr - dev

    # ── Features from your comments ───────────────────────────────────────────
    df["near_upper_band"] = close >= (lr + 0.8 * dev)   # strong bullish trend
    df["near_lower_band"] = close <= (lr - 0.8 * dev)   # strong bearish trend
    df["touches_upper"]   = close >= (lr + dev)          # stretched / overvalued
    df["touches_lower"]   = close <= (lr - dev)          # compressed / undervalued
    df["broke_above"]     = (close > (lr + dev)) & (close.shift(1) <= (lr + dev).shift(1))  # trend acceleration up
    df["broke_below"]     = (close < (lr - dev)) & (close.shift(1) >= (lr - dev).shift(1))  # trend acceleration down
    band_width            = 2 * dev / (lr.abs() + 1e-9)
    df["bands_widening"]  = band_width > band_width.shift(3)   # volatility increasing
    df["bands_narrowing"] = band_width < band_width.shift(3)   # consolidation coming

    # ── MACD Entry Confirmation ───────────────────────────────────────────────
    macd_cross_up   = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    macd_cross_down = (macd < signal) & (macd.shift(1) >= signal.shift(1))

    # MACD Bullish: MACD crosses above signal while testing / bouncing off lower band
    df["macd_bullish_entry"] = macd_cross_up & df["near_lower_band"]

    # MACD Bearish: MACD crosses below signal while touching / breaking upper band
    df["macd_bearish_entry"] = macd_cross_down & df["near_upper_band"]

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    df["rsi"]        = 100 - (100 / (1 + rs))
    df["rsi_overbought"] = df["rsi"] > 70   # stretched to the upside
    df["rsi_oversold"]   = df["rsi"] < 30   # stretched to the downside

    return df
