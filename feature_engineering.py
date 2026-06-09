"""
feature_engineering.py
===========
All feature engineering in one place. Import and call build_features(df)
with a raw OHLCV dataframe to get back the fully enriched version.
"""

import numpy as np
import pandas as pd
import ta
import yfinance as yf


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
    df["broke_above"]     = (close > (lr + dev)) & (close.shift(1) <= (lr + dev).shift(1))
    df["broke_below"]     = (close < (lr - dev)) & (close.shift(1) >= (lr - dev).shift(1))
    band_width            = 2 * dev / (lr.abs() + 1e-9)
    df["bands_widening"]  = band_width > band_width.shift(3)
    df["bands_narrowing"] = band_width < band_width.shift(3)

    # MACD Entry Confirmation
    macd_cross_up   = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    macd_cross_down = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    df["macd_bullish_entry"] = macd_cross_up   & df["near_lower_band"]
    df["macd_bearish_entry"] = macd_cross_down & df["near_upper_band"]

    # RSI
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi"]           = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))
    df["rsi_overbought"]= df["rsi"] > 70
    df["rsi_oversold"]  = df["rsi"] < 30

    return df


def _add_mean_reversion(df):
    """Mean reversion boolean indicators."""
    close = df["Close"].squeeze()

    sma20 = close.rolling(20).mean()
    df["mr_below_sma20"] = (close < sma20).astype(float)
    df["mr_above_sma20"] = (close > sma20).astype(float)

    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["mr_bb_below_lower"] = (close < bb_lower).astype(float)
    df["mr_bb_above_upper"] = (close > bb_upper).astype(float)

    if "rsi" in df.columns:
        rsi = df["rsi"]
    else:
        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi      = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))
    df["mr_rsi_oversold"]   = (rsi < 30).astype(float)
    df["mr_rsi_overbought"] = (rsi > 70).astype(float)

    z_score = (close - sma20) / std20.replace(0, np.nan)
    df["mr_z_score_low"]  = (z_score < -1.5).astype(float)
    df["mr_z_score_high"] = (z_score >  1.5).astype(float)

    typical = (df["High"] + df["Low"] + close) / 3
    vwap    = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["mr_below_vwap"] = (close < vwap).astype(float)
    df["mr_above_vwap"] = (close > vwap).astype(float)

    return df


def _add_momentum(df):
    """Momentum boolean indicators."""
    close = df["Close"].squeeze()

    roc20 = close.pct_change(20) * 100
    df["mo_roc_positive_20"] = (roc20 > 0).astype(float)
    df["mo_roc_negative_20"] = (roc20 < 0).astype(float)

    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    prev_diff = (sma50 - sma200).shift(1)
    curr_diff = (sma50 - sma200)
    df["mo_golden_cross"] = ((prev_diff < 0) & (curr_diff >= 0)).astype(float)
    df["mo_death_cross"]  = ((prev_diff > 0) & (curr_diff <= 0)).astype(float)

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

    if "ADX" in df.columns:
        df["mo_adx_trending"] = (df["ADX"] > 25).astype(float)
    else:
        adx_ind = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
        df["mo_adx_trending"] = (adx_ind.adx() > 25).astype(float)

    high20 = close.rolling(20).max().shift(1)
    low20  = close.rolling(20).min().shift(1)
    df["mo_breakout_high20"] = (close > high20).astype(float)
    df["mo_breakdown_low20"] = (close < low20).astype(float)

    vol_ma20 = df["Volume"].rolling(20).mean()
    df["mo_volume_surge"] = (df["Volume"] > 2 * vol_ma20).astype(float)

    daily_ret   = close.diff()
    up          = (daily_ret > 0).astype(int)
    down        = (daily_ret < 0).astype(int)
    consec_up   = up.groupby((up   == 0).cumsum()).cumsum()
    consec_down = down.groupby((down == 0).cumsum()).cumsum()
    df["mo_consecutive_up3"]   = (consec_up   >= 3).astype(float)
    df["mo_consecutive_down3"] = (consec_down >= 3).astype(float)

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
    """ICT-style Liquidity Sweep → Fair Value Gap (FVG) entry setups."""

    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    open_ = df["Open"].values
    n     = len(df)

    bull_fvg_top = np.full(n, np.nan)
    bull_fvg_bot = np.full(n, np.nan)
    bear_fvg_top = np.full(n, np.nan)
    bear_fvg_bot = np.full(n, np.nan)

    for i in range(2, n):
        if low[i] > high[i - 2]:
            bull_fvg_bot[i] = high[i - 2]
            bull_fvg_top[i] = low[i]
        if high[i] < low[i - 2]:
            bear_fvg_bot[i] = high[i]
            bear_fvg_top[i] = low[i - 2]

    active_bull_top = np.full(n, np.nan)
    active_bull_bot = np.full(n, np.nan)
    active_bear_top = np.full(n, np.nan)
    active_bear_bot = np.full(n, np.nan)

    cur_bull_top = cur_bull_bot = np.nan
    cur_bear_top = cur_bear_bot = np.nan

    for i in range(n):
        if not np.isnan(bull_fvg_top[i]):
            cur_bull_top = bull_fvg_top[i]
            cur_bull_bot = bull_fvg_bot[i]
        if not np.isnan(bear_fvg_top[i]):
            cur_bear_top = bear_fvg_top[i]
            cur_bear_bot = bear_fvg_bot[i]

        if not np.isnan(cur_bull_bot) and close[i] < cur_bull_bot:
            cur_bull_top = cur_bull_bot = np.nan
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

    df["in_bull_fvg"] = (
        (df["Low"]  <= df["fvg_bull_top"]) &
        (df["Close"] >= df["fvg_bull_bot"])
    ).astype(float)

    df["in_bear_fvg"] = (
        (df["High"]  >= df["fvg_bear_bot"]) &
        (df["Close"] <= df["fvg_bear_top"])
    ).astype(float)

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

    df["setup_bull_sweep_fvg"] = (df["recent_low_sweep"] * df["in_bull_fvg"]).clip(0, 1)
    df["setup_bear_sweep_fvg"] = (df["recent_high_sweep"] * df["in_bear_fvg"]).clip(0, 1)

    is_green = (df["Close"] > df["Open"]).astype(float)
    is_red   = (df["Close"] < df["Open"]).astype(float)

    df["setup_bull_confirmed"] = (df["setup_bull_sweep_fvg"] * is_green).clip(0, 1)
    df["setup_bear_confirmed"] = (df["setup_bear_sweep_fvg"] * is_red).clip(0, 1)

    return df


def _add_index_divergence(df, period="30d", interval="5m", min_ret_threshold=0.0005):
    """
    SPY / QQQ divergence features for mean-reversion stat-arb signals.

    Downloads SPY (S&P 500) and QQQ (NASDAQ-100) at the same interval as
    the primary dataframe, aligns them bar-by-bar, then computes per-bar
    returns and flags when the two indices move in opposite directions.

    A divergence between SPY and QQQ is a mean-reversion signal: historically
    the two are highly correlated (~0.95), so when they decouple it tends to
    be transient and one or both revert toward the other.

    Parameters
    ----------
    df : pd.DataFrame
        Primary OHLCV dataframe (already indexed by datetime).
    period : str
        yfinance period string matching what was used to download df.
    interval : str
        yfinance interval string matching df (e.g. "5m", "1h").
    min_ret_threshold : float
        Minimum absolute return for a bar to count as a real move rather
        than noise. Default 0.05% (0.0005). Bars where either index moves
        less than this are treated as flat and do not trigger divergence.

    Columns added
    -------------
    spy_ret             Bar-over-bar return of SPY, aligned to df index
    qqq_ret             Bar-over-bar return of QQQ, aligned to df index
    idx_div_spy_up_qqq_down   SPY up, QQQ down — bearish for tech, bullish macro
    idx_div_spy_down_qqq_up   SPY down, QQQ up — bullish for tech, bearish macro
    idx_div_any               Either divergence direction is True
    idx_div_rolling3          Divergence occurred in any of last 3 bars (persistence flag)
    idx_corr_20               20-bar rolling correlation between SPY and QQQ returns
                              (low/negative value = divergence regime is elevated)
    idx_corr_breakdown        Rolling correlation dropped below 0.5 — regime-level decoupling
    """
    # ── Download index data ───────────────────────────────────────────────────
    try:
        spy_raw = yf.download("SPY", period=period, interval=interval,
                              progress=False, auto_adjust=True)
        qqq_raw = yf.download("QQQ", period=period, interval=interval,
                              progress=False, auto_adjust=True)
    except Exception as e:
        print(f"[index_divergence] Download failed: {e}. Filling features with 0.")
        for col in ["spy_ret", "qqq_ret", "idx_div_spy_up_qqq_down",
                    "idx_div_spy_down_qqq_up", "idx_div_any",
                    "idx_div_rolling3", "idx_corr_20", "idx_corr_breakdown"]:
            df[col] = 0.0
        return df

    # Flatten MultiIndex columns if present
    for raw in (spy_raw, qqq_raw):
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

    spy_close = spy_raw["Close"].squeeze()
    qqq_close = qqq_raw["Close"].squeeze()

    # ── Bar-over-bar returns ──────────────────────────────────────────────────
    spy_ret = spy_close.pct_change()
    qqq_ret = qqq_close.pct_change()

    # ── Align to primary df index ─────────────────────────────────────────────
    # Use reindex + ffill so we handle any minor timestamp mismatches between
    # the primary ticker and SPY/QQQ (e.g. a few seconds of offset).
    spy_ret = spy_ret.reindex(df.index, method="ffill")
    qqq_ret = qqq_ret.reindex(df.index, method="ffill")

    df["spy_ret"] = spy_ret.values
    df["qqq_ret"] = qqq_ret.values

    # ── Directional divergence flags ─────────────────────────────────────────
    # A bar counts as a real move only if its absolute return exceeds the
    # threshold — this filters out flat/noise bars at open/close.
    spy_up   = (spy_ret >  min_ret_threshold)
    spy_down = (spy_ret < -min_ret_threshold)
    qqq_up   = (qqq_ret >  min_ret_threshold)
    qqq_down = (qqq_ret < -min_ret_threshold)

    # SPY up, QQQ down → macro bid but tech sold off → watch for QQQ reversion up
    df["idx_div_spy_up_qqq_down"] = (spy_up   & qqq_down).astype(float)

    # SPY down, QQQ up → tech bid but broad market sold off → watch for SPY reversion up
    # or QQQ reversion down
    df["idx_div_spy_down_qqq_up"] = (spy_down & qqq_up).astype(float)

    # Either direction
    df["idx_div_any"] = (
        df["idx_div_spy_up_qqq_down"] + df["idx_div_spy_down_qqq_up"]
    ).clip(0, 1)

    # Persistence: did divergence occur in any of the last 3 bars?
    df["idx_div_rolling3"] = (
        df["idx_div_any"]
        .rolling(3, min_periods=1)
        .max()
        .astype(float)
    )

    # ── Rolling correlation ───────────────────────────────────────────────────
    # Low or negative rolling correlation = indices are meaningfully decoupling.
    corr = spy_ret.rolling(20, min_periods=10).corr(qqq_ret)
    corr = corr.reindex(df.index, method="ffill")
    df["idx_corr_20"] = corr.values

    # Correlation breakdown: rolling corr below 0.5 flags a divergence regime
    df["idx_corr_breakdown"] = (corr < 0.5).astype(float)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, period: str = "30d", interval: str = "5m") -> pd.DataFrame:
    """
    Run all feature engineering on a raw OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV dataframe from yfinance.
    period : str
        The period used when downloading df — passed through to
        _add_index_divergence so SPY/QQQ are downloaded over the same window.
    interval : str
        The interval used when downloading df (e.g. "5m").

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
    df = _add_sweep_fvg_setups(df)
    df = _add_index_divergence(df, period=period, interval=interval)

    # Safety net: force every boolean-signal column to float32
    bool_signal_cols = [c for c in df.columns if c in FEATURES and c != "Close"]
    for col in bool_signal_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df


# Feature columns to use as model input (edit here to change for both files)
FEATURES = [
    "Close",
    # ── sweep + FVG setups ────────────────────────────────────────────────────
    # ── Mean reversion ----────────────────────────────────────────────────────
    # ── News sentiment ----────────────────────────────────────────────────────
    "news_bull",
    "news_bear",
    "news_sentiment_strong_bull",
    "news_sentiment_strong_bear",
    # ── momentum ─────────────────────────────────────────────────────────────
    # ── index divergence (stat-arb / mean reversion) ─────────────────────────
    # "idx_div_spy_up_qqq_down",   # SPY up, QQQ down — tech lagging broad market
    # "idx_div_spy_down_qqq_up",   # SPY down, QQQ up — tech leading, macro lagging
    # "idx_div_rolling3",         # divergence persisted in last 3 bars
    # "idx_corr_breakdown"         # SPY/QQQ correlation < 0.5 flags decoupling regime
]