"""
feature_engineering.py
===========
All feature engineering in one place. Import and call build_features(df)
with a raw OHLCV dataframe to get back the fully enriched version.
"""

import numpy as np
import pandas as pd
import ta


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _rolling_linreg_endpoint(series, length):
    """Matches Pine Script's ta.linreg(close, length, 0)."""
    y      = series.values.astype(float)
    result = np.full(len(y), np.nan)
    x      = np.arange(length, dtype=float)
    xm     = x.mean()
    for i in range(length - 1, len(y)):
        yw        = y[i - length + 1 : i + 1]
        ym        = yw.mean()
        slope     = np.dot(x - xm, yw - ym) / (np.dot(x - xm, x - xm) + 1e-12)
        result[i] = ym + slope * (length - 1 - xm)
    return pd.Series(result, index=series.index)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS
# ─────────────────────────────────────────────────────────────────────────────

def _add_liquidity_sweeps(df):
    """Previous day high/low wick sweeps."""
    daily = df.groupby(df.index.date).agg(
        day_high=("High", "max"),
        day_low =("Low",  "min"),
    )
    daily.index          = pd.to_datetime(daily.index)
    daily["prev_day_high"] = daily["day_high"].shift(1)
    daily["prev_day_low"]  = daily["day_low"].shift(1)

    df["prev_day_high"]    = pd.to_numeric(df.index.normalize().map(daily["prev_day_high"]), errors="coerce")
    df["prev_day_low"]     = pd.to_numeric(df.index.normalize().map(daily["prev_day_low"]),  errors="coerce")
    df["high_wick_sweep"]  = (df["High"] > df["prev_day_high"]) & (df["Close"] < df["prev_day_high"])
    df["low_wick_sweep"]   = (df["Low"]  < df["prev_day_low"])  & (df["Close"] > df["prev_day_low"])
    return df


def _add_candle_features(df):
    """Basic candle direction features."""
    df["isGreen"] = df["Close"] > df["Open"]
    df["isHigh"]  = df["isGreen"] & (df["Close"] > df["Close"].shift(1))
    df["isLow"]   = ~df["isGreen"] & (df["Close"] < df["Close"].shift(1))
    return df


def _add_structural_trend(df, lookback=50, swing_window=5):
    """Swing-point based trend classification."""

    def find_swing_points(highs, lows, window):
        swings = []
        n = len(highs)
        for i in range(window, n - window):
            if highs.iloc[i] == highs.iloc[i - window : i + window + 1].max():
                swings.append((i, "H", highs.iloc[i]))
            if lows.iloc[i] == lows.iloc[i - window : i + window + 1].min():
                swings.append((i, "L", lows.iloc[i]))
        swings.sort(key=lambda x: x[0])
        alternated = []
        for s in swings:
            if not alternated:
                alternated.append(s)
            elif s[1] == alternated[-1][1]:
                if s[1] == "H" and s[2] >= alternated[-1][2]: alternated[-1] = s
                elif s[1] == "L" and s[2] <= alternated[-1][2]: alternated[-1] = s
            else:
                alternated.append(s)
        return alternated

    def classify_trend(swings):
        if len(swings) < 3: return "ranging"
        recent = swings[-4:]
        types  = [s[1] for s in recent]
        vals   = [s[2] for s in recent]
        if types[-3:] == ["H", "L", "H"]:
            if vals[-1] > vals[-3]: return "uptrend"
            if vals[-1] < vals[-3]: return "downtrend"
        if types[-3:] == ["L", "H", "L"]:
            if vals[-1] > vals[-3]: return "uptrend"
            if vals[-1] < vals[-3]: return "downtrend"
        if len(recent) == 4:
            highs = [s[2] for s in recent if s[1] == "H"]
            lows  = [s[2] for s in recent if s[1] == "L"]
            if len(highs) >= 2 and len(lows) >= 2:
                if highs[-1] > highs[-2] and lows[-1] > lows[-2]: return "uptrend"
                if highs[-1] < highs[-2] and lows[-1] < lows[-2]: return "downtrend"
        return "ranging"

    trends = ["ranging"] * len(df)
    for i in range(lookback, len(df)):
        w         = df.iloc[i - lookback : i + 1]
        swings    = find_swing_points(w["High"], w["Low"], swing_window)
        trends[i] = classify_trend(swings)

    df["structural_trend"] = trends
    dummies = pd.get_dummies(df["structural_trend"], prefix="trend", drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    return df


def _add_adx(df):
    """ADX trend strength features."""
    adx_ind = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["ADX"] = adx_ind.adx()
    df["DMP"] = adx_ind.adx_pos()
    df["DMN"] = adx_ind.adx_neg()

    df["is_trending"]  = df["ADX"] > 25
    df["adx_uptrend"]  = (df["ADX"] > 25) & (df["DMP"] > df["DMN"])
    df["adx_downtrend"]= (df["ADX"] > 25) & (df["DMN"] > df["DMP"])

    df["DI_cross_up"]  = (df["DMP"] > df["DMN"]) & (df["DMP"].shift(1) < df["DMN"].shift(1))
    df["DI_cross_down"]= (df["DMP"] < df["DMN"]) & (df["DMP"].shift(1) > df["DMN"].shift(1))

    df["ADX_slope"]          = df["ADX"].diff(3)
    df["trend_strengthening"]= df["ADX_slope"] > 0
    df["trend_weakening"]    = df["ADX_slope"] < 0

    df["strong_up"]   = df["adx_uptrend"]   & df["trend_strengthening"]
    df["fading_up"]   = df["adx_uptrend"]   & df["trend_weakening"]
    df["strong_down"] = df["adx_downtrend"] & df["trend_strengthening"]
    df["fading_down"] = df["adx_downtrend"] & df["trend_weakening"]
    return df


def _add_macd_lr(df, fast=12, slow=26, signal_len=9, lr_length=100, lr_mult=4.0):
    """MACD + Linear Regression Channel features."""
    close = df["Close"].squeeze()

    # MACD
    macd   = _ema(close, fast) - _ema(close, slow)
    signal = _ema(macd, signal_len)
    df["macd_line"]   = macd
    df["macd_signal"] = signal
    df["macd_hist"]   = macd - signal

    # Linear Regression Channel
    lr  = _rolling_linreg_endpoint(close, lr_length)
    dev = lr_mult * (close - lr).rolling(lr_length).std()
    df["lr_upper"] = lr + dev
    df["lr_lower"] = lr - dev

    # Band position features (from Pine Script comments)
    df["near_upper_band"] = close >= (lr + 0.8 * dev)   # strong bullish trend
    df["near_lower_band"] = close <= (lr - 0.8 * dev)   # strong bearish trend
    df["touches_upper"]   = close >= (lr + dev)          # stretched / overvalued
    df["touches_lower"]   = close <= (lr - dev)          # compressed / undervalued
    df["broke_above"]     = (close > (lr + dev)) & (close.shift(1) <= (lr + dev).shift(1))  # trend acceleration up
    df["broke_below"]     = (close < (lr - dev)) & (close.shift(1) >= (lr - dev).shift(1))  # trend acceleration down
    band_width            = 2 * dev / (lr.abs() + 1e-9)
    df["bands_widening"]  = band_width > band_width.shift(3)   # volatility increasing
    df["bands_narrowing"] = band_width < band_width.shift(3)   # consolidation coming

    # MACD Entry Confirmation
    macd_cross_up   = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    macd_cross_down = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    df["macd_bullish_entry"] = macd_cross_up   & df["near_lower_band"]  # MACD bullish entry
    df["macd_bearish_entry"] = macd_cross_down & df["near_upper_band"]  # MACD bearish entry

    # RSI
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi"]           = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))
    df["rsi_overbought"]= df["rsi"] > 70   # stretched to the upside
    df["rsi_oversold"]  = df["rsi"] < 30   # stretched to the downside

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering on a raw OHLCV dataframe.
    Returns the enriched dataframe.
    """
    df = df.copy()
    df = _add_liquidity_sweeps(df)
    df = _add_candle_features(df)
    df = _add_structural_trend(df)
    df = _add_adx(df)
    df = _add_macd_lr(df)
    return df


# Feature columns to use as model input (edit here to change for both files)
FEATURES = [
    "Close",
    "macd_bullish_entry",
    "macd_bearish_entry",
]