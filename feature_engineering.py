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


def _add_mean_reversion(df):
    """
    Mean reversion boolean indicators.

    Columns added
    -------------
    mr_below_sma20          Price below 20-day SMA (potential long reversion)
    mr_above_sma20          Price above 20-day SMA (potential short reversion)
    mr_bb_below_lower       Price below lower Bollinger Band (20d, 2sigma) — oversold stretch
    mr_bb_above_upper       Price above upper Bollinger Band (20d, 2sigma) — overbought stretch
    mr_rsi_oversold         RSI-14 < 30 — stretched to the downside
    mr_rsi_overbought       RSI-14 > 70 — stretched to the upside
    mr_z_score_low          20-day price Z-score < -1.5 — statistically cheap
    mr_z_score_high         20-day price Z-score >  1.5 — statistically expensive
    mr_below_vwap           Close below rolling VWAP approximation
    mr_above_vwap           Close above rolling VWAP approximation
    """
    close = df["Close"].squeeze()

    # SMA-20
    sma20 = close.rolling(20).mean()
    df["mr_below_sma20"] = (close < sma20).astype(bool)
    df["mr_above_sma20"] = (close > sma20).astype(bool)

    # Bollinger Bands (20d, 2sigma)
    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["mr_bb_below_lower"] = (close < bb_lower).astype(bool)
    df["mr_bb_above_upper"] = (close > bb_upper).astype(bool)

    # RSI-14 (reuse existing "rsi" column if already computed, else compute fresh)
    if "rsi" in df.columns:
        rsi = df["rsi"]
    else:
        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi      = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))
    df["mr_rsi_oversold"]   = (rsi < 30).astype(bool)
    df["mr_rsi_overbought"] = (rsi > 70).astype(bool)

    # Z-Score (20-day rolling)
    z_score = (close - sma20) / std20.replace(0, np.nan)
    df["mr_z_score_low"]  = (z_score < -1.5).astype(bool)
    df["mr_z_score_high"] = (z_score >  1.5).astype(bool)

    # VWAP approximation (typical price x volume, cumulative within each day)
    typical = (df["High"] + df["Low"] + close) / 3
    vwap    = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["mr_below_vwap"] = (close < vwap).astype(bool)
    df["mr_above_vwap"] = (close > vwap).astype(bool)

    return df


def _add_momentum(df):
    """
    Momentum boolean indicators.

    Columns added
    -------------
    mo_roc_positive_20          20-day Rate-of-Change > 0 — upward momentum
    mo_roc_negative_20          20-day Rate-of-Change < 0 — downward momentum
    mo_golden_cross             SMA-50 crossed above SMA-200 — major bullish signal
    mo_death_cross              SMA-50 crossed below SMA-200 — major bearish signal
    mo_macd_cross_up            MACD line crossed above signal line (standalone, no band filter)
    mo_macd_cross_down          MACD line crossed below signal line (standalone, no band filter)
    mo_adx_trending             ADX-14 > 25 — trend strong enough to trade
    mo_breakout_high20          Close > 20-day highest high — upside breakout
    mo_breakdown_low20          Close < 20-day lowest low  — downside breakdown
    mo_volume_surge             Volume > 2x its 20-day average — momentum confirmation
    mo_consecutive_up3          3+ consecutive higher closes
    mo_consecutive_down3        3+ consecutive lower closes
    mo_combo_long               >= 2 momentum long signals firing together (high conviction)
    mo_combo_short              >= 2 momentum short signals firing together (high conviction)
    """
    close = df["Close"].squeeze()

    # Rate of Change (20d)
    roc20 = close.pct_change(20) * 100
    df["mo_roc_positive_20"] = (roc20 > 0).astype(bool)
    df["mo_roc_negative_20"] = (roc20 < 0).astype(bool)

    # Golden / Death Cross
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    prev_diff = (sma50 - sma200).shift(1)
    curr_diff = (sma50 - sma200)
    df["mo_golden_cross"] = ((prev_diff < 0) & (curr_diff >= 0)).astype(bool)
    df["mo_death_cross"]  = ((prev_diff > 0) & (curr_diff <= 0)).astype(bool)

    # MACD crossovers (standalone — no band filter, unlike macd_bullish/bearish_entry)
    if "macd_line" in df.columns and "macd_signal" in df.columns:
        macd   = df["macd_line"]
        signal = df["macd_signal"]
    else:
        macd   = _ema(close, 12) - _ema(close, 26)
        signal = _ema(macd, 9)
    prev_md = (macd - signal).shift(1)
    curr_md = (macd - signal)
    df["mo_macd_cross_up"]   = ((prev_md < 0) & (curr_md >= 0)).astype(bool)
    df["mo_macd_cross_down"] = ((prev_md > 0) & (curr_md <= 0)).astype(bool)

    # ADX trending (reuse existing ADX column if available)
    if "ADX" in df.columns:
        df["mo_adx_trending"] = (df["ADX"] > 25).astype(bool)
    else:
        adx_ind = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
        df["mo_adx_trending"] = (adx_ind.adx() > 25).astype(bool)

    # 20-day breakout / breakdown (exclude today's bar)
    high20 = close.rolling(20).max().shift(1)
    low20  = close.rolling(20).min().shift(1)
    df["mo_breakout_high20"] = (close > high20).astype(bool)
    df["mo_breakdown_low20"] = (close < low20).astype(bool)

    # Volume surge
    vol_ma20 = df["Volume"].rolling(20).mean()
    df["mo_volume_surge"] = (df["Volume"] > 2 * vol_ma20).astype(bool)

    # Consecutive up / down closes
    daily_ret   = close.diff()
    up          = (daily_ret > 0).astype(int)
    down        = (daily_ret < 0).astype(int)
    consec_up   = up.groupby((up   == 0).cumsum()).cumsum()
    consec_down = down.groupby((down == 0).cumsum()).cumsum()
    df["mo_consecutive_up3"]   = (consec_up   >= 3).astype(bool)
    df["mo_consecutive_down3"] = (consec_down >= 3).astype(bool)

    # High-conviction composite signals (>= 2 signals firing together)
    long_votes = (
        df["mo_roc_positive_20"].astype(int) +
        df["mo_macd_cross_up"].astype(int) +
        df["mo_breakout_high20"].astype(int) +
        df["mo_volume_surge"].astype(int) +
        df["mo_consecutive_up3"].astype(int)
    )
    short_votes = (
        df["mo_roc_negative_20"].astype(int) +
        df["mo_macd_cross_down"].astype(int) +
        df["mo_breakdown_low20"].astype(int) +
        df["mo_volume_surge"].astype(int) +
        df["mo_consecutive_down3"].astype(int)
    )
    df["mo_combo_long"]  = (long_votes  >= 2).astype(bool)
    df["mo_combo_short"] = (short_votes >= 2).astype(bool)

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
    df = _add_mean_reversion(df)
    df = _add_momentum(df)
    return df


# Feature columns to use as model input (edit here to change for both files)
FEATURES = [
    "Close",
    # ── existing entries ──────────────────────────────────────────────────────
    "macd_bullish_entry",
    "macd_bearish_entry",
    # ── mean reversion ────────────────────────────────────────────────────────
    "mr_below_sma20",
    "mr_above_sma20",
    "mr_bb_below_lower",
    "mr_bb_above_upper",
    "mr_rsi_oversold",
    "mr_rsi_overbought",
    "mr_z_score_low",
    "mr_z_score_high",
    "mr_below_vwap",
    "mr_above_vwap",
    # ── momentum ──────────────────────────────────────────────────────────────
    "mo_roc_positive_20",
    "mo_roc_negative_20",
    "mo_golden_cross",
    "mo_death_cross",
    "mo_macd_cross_up",
    "mo_macd_cross_down",
    "mo_adx_trending",
    "mo_breakout_high20",
    "mo_breakdown_low20",
    "mo_volume_surge",
    "mo_consecutive_up3",
    "mo_consecutive_down3",
    "mo_combo_long",
    "mo_combo_short",
]