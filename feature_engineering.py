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
    df["mr_below_sma20"] = (close < sma20).astype(float)
    df["mr_above_sma20"] = (close > sma20).astype(float)

    # Bollinger Bands (20d, 2sigma)
    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["mr_bb_below_lower"] = (close < bb_lower).astype(float)
    df["mr_bb_above_upper"] = (close > bb_upper).astype(float)

    # RSI-14 (reuse existing "rsi" column if already computed, else compute fresh)
    if "rsi" in df.columns:
        rsi = df["rsi"]
    else:
        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi      = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))
    df["mr_rsi_oversold"]   = (rsi < 30).astype(float)
    df["mr_rsi_overbought"] = (rsi > 70).astype(float)

    # Z-Score (20-day rolling)
    z_score = (close - sma20) / std20.replace(0, np.nan)
    df["mr_z_score_low"]  = (z_score < -1.5).astype(float)
    df["mr_z_score_high"] = (z_score >  1.5).astype(float)

    # VWAP approximation (typical price x volume, cumulative within each day)
    typical = (df["High"] + df["Low"] + close) / 3
    vwap    = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["mr_below_vwap"] = (close < vwap).astype(float)
    df["mr_above_vwap"] = (close > vwap).astype(float)

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
    df["mo_roc_positive_20"] = (roc20 > 0).astype(float)
    df["mo_roc_negative_20"] = (roc20 < 0).astype(float)

    # Golden / Death Cross
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    prev_diff = (sma50 - sma200).shift(1)
    curr_diff = (sma50 - sma200)
    df["mo_golden_cross"] = ((prev_diff < 0) & (curr_diff >= 0)).astype(float)
    df["mo_death_cross"]  = ((prev_diff > 0) & (curr_diff <= 0)).astype(float)

    # MACD crossovers (standalone — no band filter, unlike macd_bullish/bearish_entry)
    if "macd_line" in df.columns and "macd_signal" in df.columns:
        macd   = df["macd_line"]
        signal = df["macd_signal"]
    else:
        macd   = _ema(close, 12) - _ema(close, 26)
        signal = _ema(macd, 9)
    prev_md = (macd - signal).shift(1)
    curr_md = (macd - signal)
    df["mo_macd_cross_up"]   = ((prev_md < 0) & (curr_md >= 0)).astype(float)
    df["mo_macd_cross_down"] = ((prev_md > 0) & (curr_md <= 0)).astype(float)

    # ADX trending (reuse existing ADX column if available)
    if "ADX" in df.columns:
        df["mo_adx_trending"] = (df["ADX"] > 25).astype(float)
    else:
        adx_ind = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
        df["mo_adx_trending"] = (adx_ind.adx() > 25).astype(float)

    # 20-day breakout / breakdown (exclude today's bar)
    high20 = close.rolling(20).max().shift(1)
    low20  = close.rolling(20).min().shift(1)
    df["mo_breakout_high20"] = (close > high20).astype(float)
    df["mo_breakdown_low20"] = (close < low20).astype(float)

    # Volume surge
    vol_ma20 = df["Volume"].rolling(20).mean()
    df["mo_volume_surge"] = (df["Volume"] > 2 * vol_ma20).astype(float)

    # Consecutive up / down closes
    daily_ret   = close.diff()
    up          = (daily_ret > 0).astype(int)
    down        = (daily_ret < 0).astype(int)
    consec_up   = up.groupby((up   == 0).cumsum()).cumsum()
    consec_down = down.groupby((down == 0).cumsum()).cumsum()
    df["mo_consecutive_up3"]   = (consec_up   >= 3).astype(float)
    df["mo_consecutive_down3"] = (consec_down >= 3).astype(float)

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
    df["mo_combo_long"]  = (long_votes  >= 2).astype(float)
    df["mo_combo_short"] = (short_votes >= 2).astype(float)

    return df


def _add_sweep_fvg_setups(df, sweep_lookback=15, fvg_lookback=30):
    """
    ICT-style Liquidity Sweep → Fair Value Gap (FVG) entry setups.

    Logic overview
    --------------
    A BULLISH setup requires two conditions to both be true within a
    recent window:
      1. SWEEP LOW  — within the last `sweep_lookback` candles, price
         wicked below a prior session low and closed back above it
         (stop-hunt of sell-side liquidity).
      2. BULLISH FVG ENTRY — the current candle's low trades into a
         bullish FVG (gap between candle[i-2].high and candle[i].low
         where candle[i-1] is a strong up-move) that formed AFTER the
         sweep, signalling smart-money stepped in to fill imbalance.

    A BEARISH setup is the mirror:
      1. SWEEP HIGH — price wicked above a prior session high and
         closed back below it (stop-hunt of buy-side liquidity).
      2. BEARISH FVG ENTRY — the current candle's high trades into a
         bearish FVG (gap between candle[i-2].low and candle[i].high
         where candle[i-1] is a strong down-move) that formed AFTER
         the sweep.

    Columns added
    -------------
    fvg_bull_top            Upper boundary of the most recent active bullish FVG
    fvg_bull_bot            Lower boundary of the most recent active bullish FVG
    fvg_bear_top            Upper boundary of the most recent active bearish FVG
    fvg_bear_bot            Lower boundary of the most recent active bearish FVG
    in_bull_fvg             Close is inside an active bullish FVG right now
    in_bear_fvg             Close is inside an active bearish FVG right now
    recent_low_sweep        A prior-session low was swept (wicked & closed above)
                            within the last `sweep_lookback` candles
    recent_high_sweep       A prior-session high was swept (wicked & closed below)
                            within the last `sweep_lookback` candles
    setup_bull_sweep_fvg    Full bullish setup: recent low sweep + price in bullish FVG
    setup_bear_sweep_fvg    Full bearish setup: recent high sweep + price in bearish FVG
    setup_bull_confirmed    setup_bull_sweep_fvg + current candle is green (confirmation)
    setup_bear_confirmed    setup_bear_sweep_fvg + current candle is red  (confirmation)
    """

    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    open_ = df["Open"].values
    n     = len(df)

    # ── 1. Identify all Fair Value Gaps ──────────────────────────────────────
    # Bullish FVG: gap between candle[i-2].high and candle[i].low
    #   formed when candle[i-1] is a strong bullish impulse candle
    # Bearish FVG: gap between candle[i-2].low  and candle[i].high
    #   formed when candle[i-1] is a strong bearish impulse candle

    bull_fvg_top = np.full(n, np.nan)   # upper edge of bullish FVG at formation bar i
    bull_fvg_bot = np.full(n, np.nan)   # lower edge
    bear_fvg_top = np.full(n, np.nan)
    bear_fvg_bot = np.full(n, np.nan)

    for i in range(2, n):
        # Bullish FVG: candle[i].low > candle[i-2].high  (gap above)
        if low[i] > high[i - 2]:
            bull_fvg_bot[i] = high[i - 2]
            bull_fvg_top[i] = low[i]

        # Bearish FVG: candle[i].high < candle[i-2].low  (gap below)
        if high[i] < low[i - 2]:
            bear_fvg_bot[i] = high[i]
            bear_fvg_top[i] = low[i - 2]

    # ── 2. Track the most recent ACTIVE FVG at each bar ──────────────────────
    # An FVG is "active" until price fully closes through it.
    # We carry the most recently formed FVG forward until it is invalidated.

    active_bull_top = np.full(n, np.nan)
    active_bull_bot = np.full(n, np.nan)
    active_bear_top = np.full(n, np.nan)
    active_bear_bot = np.full(n, np.nan)

    cur_bull_top = cur_bull_bot = np.nan
    cur_bear_top = cur_bear_bot = np.nan

    for i in range(n):
        # New FVG formed this bar → update active
        if not np.isnan(bull_fvg_top[i]):
            cur_bull_top = bull_fvg_top[i]
            cur_bull_bot = bull_fvg_bot[i]
        if not np.isnan(bear_fvg_top[i]):
            cur_bear_top = bear_fvg_top[i]
            cur_bear_bot = bear_fvg_bot[i]

        # Invalidate bullish FVG if price closes below its bottom
        if not np.isnan(cur_bull_bot) and close[i] < cur_bull_bot:
            cur_bull_top = cur_bull_bot = np.nan

        # Invalidate bearish FVG if price closes above its top
        if not np.isnan(cur_bear_top) and close[i] > cur_bear_top:
            cur_bear_top = cur_bear_bot = np.nan

        active_bull_top[i] = cur_bull_top
        active_bull_bot[i] = cur_bull_bot
        active_bear_top[i] = cur_bear_top
        active_bear_bot[i] = cur_bear_bot

    df["fvg_bull_top"] = active_bull_top
    df["fvg_bull_bot"] = active_bull_bot
    df["fvg_bear_top"] = active_bear_top
    df["fvg_bear_bot"] = active_bear_bot

    # ── 3. Is price currently inside an FVG? ─────────────────────────────────
    # Bullish FVG entry: low trades into the gap (low <= top, close >= bot)
    df["in_bull_fvg"] = (
        (df["Low"]  <= df["fvg_bull_top"]) &
        (df["Close"] >= df["fvg_bull_bot"])
    ).astype(float)

    # Bearish FVG entry: high trades into the gap (high >= bot, close <= top)
    df["in_bear_fvg"] = (
        (df["High"]  >= df["fvg_bear_bot"]) &
        (df["Close"] <= df["fvg_bear_top"])
    ).astype(float)

    # ── 4. Recent sweep flags (within last N candles) ─────────────────────────
    # Reuse the per-bar sweep booleans already added by _add_liquidity_sweeps.
    # Roll a window to check if a sweep occurred in the last sweep_lookback bars.
    if "low_wick_sweep" not in df.columns or "high_wick_sweep" not in df.columns:
        raise RuntimeError(
            "_add_sweep_fvg_setups requires _add_liquidity_sweeps to run first."
        )

    df["recent_low_sweep"] = (
        df["low_wick_sweep"]
        .astype(int)
        .rolling(sweep_lookback, min_periods=1)
        .max()
        .astype(float)
    )
    df["recent_high_sweep"] = (
        df["high_wick_sweep"]
        .astype(int)
        .rolling(sweep_lookback, min_periods=1)
        .max()
        .astype(float)
    )

    # ── 5. Full setups ────────────────────────────────────────────────────────
    # BULLISH: prior session low swept recently + price now entering bullish FVG
    df["setup_bull_sweep_fvg"] = (
        df["recent_low_sweep"] == 1 & df["in_bull_fvg"] == 1
    ).astype(float)

    # BEARISH: prior session high swept recently + price now entering bearish FVG
    df["setup_bear_sweep_fvg"] = (
        df["recent_high_sweep"] == 1 & df["in_bear_fvg"] == 1
    ).astype(float)

    # ── 6. Confirmation candle ────────────────────────────────────────────────
    # Require the entry candle itself to close in the expected direction
    is_green = df["Close"] > df["Open"]
    is_red   = df["Close"] < df["Open"]

    df["setup_bull_confirmed"] = (df["setup_bull_sweep_fvg"] == 1 & is_green).astype(float)
    df["setup_bear_confirmed"] = (df["setup_bear_sweep_fvg"] == 1 & is_red).astype(float)

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
    df = _add_liquidity_sweeps(df)       # must run before _add_sweep_fvg_setups
    df = _add_candle_features(df)
    df = _add_structural_trend(df)
    df = _add_adx(df)
    df = _add_macd_lr(df)
    df = _add_mean_reversion(df)
    df = _add_momentum(df)
    df = _add_sweep_fvg_setups(df)
    return df


# Feature columns to use as model input (edit here to change for both files)
FEATURES = [
    "Close",
    # ── existing entries ──────────────────────────────────────────────────────
    "macd_bullish_entry",
    "macd_bearish_entry",
    # ── mean reversion ────────────────────────────────────────────────────────
    "mr_below_vwap",
    "mr_above_vwap",
    # ── momentum ──────────────────────────────────────────────────────────────
    "mo_combo_long",
    "mo_combo_short",
    # ── sweep + FVG setups ────────────────────────────────────────────────────
    "setup_bull_confirmed",
    "setup_bear_confirmed",
]