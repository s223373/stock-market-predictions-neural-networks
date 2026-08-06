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

def _add_fvg_stack_features(
    df: pd.DataFrame,
    stack_lookback: int = 50,
    target_expire_bars: int = 30,
) -> pd.DataFrame:
    """
    ICT FVG Stack and Sequential Inversion Targeting.

    Concept
    -------
    After an impulse move, multiple FVGs stack in the direction of that
    move.  When price retraces into the NEAREST FVG and *inverses* it
    (close breaks through the zone's bottom/top, flipping demand ↔ supply),
    it strongly signals continuation of the retrace toward the NEXT FVG in
    the stack.

    This function maintains an ordered stack of ALL active FVGs (vs the
    single-FVG tracking in _add_sweep_fvg_setups) and emits inversion and
    sequential-targeting signals.

    Ordering convention
    -------------------
    Bull FVGs (demand zones from up-impulse gaps) sorted DESCENDING by bot
    → highest demand zone = nearest when price retraces downward.
    Bear FVGs (supply zones from down-impulse gaps) sorted ASCENDING by bot
    → lowest supply zone = nearest when price retraces upward.

    Touched / Inversion / Arrival rules
    ------------------------------------
    - "Touched" = any bar *after creation* whose wick enters the zone
      (low <= fvg_top for bull; high >= fvg_bot for bear).
    - "Inversion" = touched FVG broken on a close basis
      (close < fvg_bot for bull; close > fvg_top for bear).
    - "Arrival" = during targeting state, wick enters the target zone AND
      close holds inside it (l <= tgt_top and c >= tgt_bot for bull;
      h >= tgt_bot and c <= tgt_top for bear).
    - Targeting fires with a one-bar lag after inversion so the signal is
      strictly forward-looking.
    - Targeting auto-resets after target_expire_bars or on arrival.

    FVG expiry
    ----------
    A FVG is removed when (a) close breaks through its bot/top (inversed),
    or (b) it is older than stack_lookback bars.

    Parameters
    ----------
    df                 : OHLCV DataFrame with DatetimeIndex.
    stack_lookback     : Max age (bars) of FVGs kept in the stack.
    target_expire_bars : Bars before a targeting state auto-resets.

    Columns added
    -------------
    fvg_bull_stack_count     # active bull FVGs remaining in stack
    fvg_bear_stack_count     # active bear FVGs remaining in stack
    fvg_impulse_up_count     bull FVGs created in last stack_lookback bars
    fvg_impulse_dn_count     bear FVGs created in last stack_lookback bars

    fvg_near_bull_top        price: top of nearest active bull FVG
    fvg_near_bull_bot        price: bottom of nearest active bull FVG
    fvg_near_bull_size       gap width (top − bot) of nearest bull FVG
    fvg_near_bull_mid        midpoint of nearest bull FVG
    fvg_near_bear_top        price: top of nearest active bear FVG
    fvg_near_bear_bot        price: bottom of nearest active bear FVG
    fvg_near_bear_size       gap width of nearest bear FVG
    fvg_near_bear_mid        midpoint of nearest bear FVG

    fvg_next_bull_top        price: top of the 2nd bull FVG in the stack
    fvg_next_bull_bot        price: bottom of the 2nd bull FVG
    fvg_next_bear_top        price: top of the 2nd bear FVG in the stack
    fvg_next_bear_bot        price: bottom of the 2nd bear FVG

    fvg_target_bull_top      price: top of the FVG currently targeted (bull)
    fvg_target_bull_bot      price: bottom of the targeted bull FVG
    fvg_target_bear_top      price: top of the FVG currently targeted (bear)
    fvg_target_bear_bot      price: bottom of the targeted bear FVG

    fvg_bull_inversion       1 when nearest bull FVG inversed (close < bot)
    fvg_bear_inversion       1 when nearest bear FVG inversed (close > top)
    fvg_bull_stacking        1 if ≥2 active bull FVGs (impulse regime)
    fvg_bear_stacking        1 if ≥2 active bear FVGs

    fvg_targeting_next_bull  1 while in post-bull-inversion targeting state
    fvg_targeting_next_bear  1 while in post-bear-inversion targeting state
    fvg_at_next_bull         1 when price first arrives at next bull FVG target
    fvg_at_next_bear         1 when price first arrives at next bear FVG target

    fvg_dist_to_near_bull    (close − near_bull_top) / close; neg = in/below zone
    fvg_dist_to_near_bear    (near_bear_bot − close) / close; neg = in/above zone
    fvg_dist_to_target_bull  normalized distance to active bull target (NaN if inactive)
    fvg_dist_to_target_bear  normalized distance to active bear target

    fvg_bull_inv_with_target 1 if inversion fired AND a next target FVG exists
    fvg_bear_inv_with_target 1 if inversion fired AND a next target FVG exists
    """
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    n     = len(df)

    # ── Pre-allocate output arrays ────────────────────────────────────────────
    bull_stack_count    = np.zeros(n)
    bear_stack_count    = np.zeros(n)
    bull_near_top       = np.full(n, np.nan)
    bull_near_bot       = np.full(n, np.nan)
    bull_near_size      = np.full(n, np.nan)
    bull_near_mid       = np.full(n, np.nan)
    bear_near_top       = np.full(n, np.nan)
    bear_near_bot       = np.full(n, np.nan)
    bear_near_size      = np.full(n, np.nan)
    bear_near_mid       = np.full(n, np.nan)
    bull_next_top       = np.full(n, np.nan)
    bull_next_bot       = np.full(n, np.nan)
    bear_next_top       = np.full(n, np.nan)
    bear_next_bot       = np.full(n, np.nan)
    bull_tgt_top_arr    = np.full(n, np.nan)
    bull_tgt_bot_arr    = np.full(n, np.nan)
    bear_tgt_top_arr    = np.full(n, np.nan)
    bear_tgt_bot_arr    = np.full(n, np.nan)
    bull_inversion      = np.zeros(n)
    bear_inversion      = np.zeros(n)
    bull_stacking       = np.zeros(n)
    bear_stacking       = np.zeros(n)
    bull_targeting      = np.zeros(n)
    bear_targeting      = np.zeros(n)
    bull_at_next        = np.zeros(n)
    bear_at_next        = np.zeros(n)
    dist_near_bull      = np.full(n, np.nan)
    dist_near_bear      = np.full(n, np.nan)
    dist_tgt_bull       = np.full(n, np.nan)
    dist_tgt_bear       = np.full(n, np.nan)
    bull_inv_with_tgt   = np.zeros(n)
    bear_inv_with_tgt   = np.zeros(n)

    # ── Impulse strength: rolling FVG creation rate ───────────────────────────
    bull_created = np.zeros(n)
    bear_created = np.zeros(n)
    for i in range(2, n):
        if low[i]  > high[i - 2]: bull_created[i] = 1.0
        if high[i] < low[i - 2]:  bear_created[i] = 1.0
    impulse_up = pd.Series(bull_created).rolling(stack_lookback, min_periods=1).sum().values
    impulse_dn = pd.Series(bear_created).rolling(stack_lookback, min_periods=1).sum().values

    # ── Active FVG stacks ─────────────────────────────────────────────────────
    # Each entry: dict {top, bot, touched (bool), created (int)}
    active_bull: list = []   # sorted DESCENDING by bot (highest demand zone first)
    active_bear: list = []   # sorted ASCENDING  by bot (lowest supply zone first)

    # Targeting state
    bull_tgt_on  = False;  bull_tgt_top = bull_tgt_bot = np.nan;  bull_tgt_at = -9999
    bear_tgt_on  = False;  bear_tgt_top = bear_tgt_bot = np.nan;  bear_tgt_at = -9999

    for i in range(n):
        c = close[i];  h = high[i];  l = low[i]

        # ── 1. Create new FVGs ─────────────────────────────────────────────
        if i >= 2:
            if l > high[i - 2]:                          # bullish gap (up-impulse)
                active_bull.append({"top": l, "bot": high[i - 2],
                                    "touched": False, "created": i})
            if h < low[i - 2]:                           # bearish gap (down-impulse)
                active_bear.append({"top": low[i - 2], "bot": h,
                                    "touched": False, "created": i})

        # ── 2. Expire stale FVGs ───────────────────────────────────────────
        active_bull = [f for f in active_bull if (i - f["created"]) <= stack_lookback]
        active_bear = [f for f in active_bear if (i - f["created"]) <= stack_lookback]

        # ── 3. Sort stacks by price proximity ─────────────────────────────
        active_bull.sort(key=lambda x: x["bot"], reverse=True)  # highest first
        active_bear.sort(key=lambda x: x["bot"])                 # lowest first

        # ── 4. Mark zones as touched (wick entry, bars after creation) ────
        # Exclude creation bar to prevent same-bar self-touch: a bull FVG's
        # top == low[creation_bar], so l <= top would trivially fire.
        for f in active_bull:
            if i > f["created"] and l <= f["top"]:
                f["touched"] = True
        for f in active_bear:
            if i > f["created"] and h >= f["bot"]:
                f["touched"] = True

        # ── 5. Detect inversions (BEFORE pruning) ─────────────────────────
        # Bull inversion: a touched demand zone broken by close below its bottom
        if any(f["touched"] and c < f["bot"] for f in active_bull):
            bull_inversion[i] = 1.0
            # Next target = highest demand zone still intact (bot <= c)
            survivors = [g for g in active_bull if c >= g["bot"]]
            if survivors:
                bull_tgt_on  = True
                bull_tgt_top = survivors[0]["top"]   # sorted desc → highest survivor
                bull_tgt_bot = survivors[0]["bot"]
                bull_tgt_at  = i
                bull_inv_with_tgt[i] = 1.0
            else:
                bull_tgt_on = False                  # inversion with no next FVG

        # Bear inversion: a touched supply zone broken by close above its top
        if any(f["touched"] and c > f["top"] for f in active_bear):
            bear_inversion[i] = 1.0
            # Next target = lowest supply zone still intact (top >= c)
            survivors = [g for g in active_bear if c <= g["top"]]
            if survivors:
                bear_tgt_on  = True
                bear_tgt_top = survivors[0]["top"]   # sorted asc → lowest survivor
                bear_tgt_bot = survivors[0]["bot"]
                bear_tgt_at  = i
                bear_inv_with_tgt[i] = 1.0
            else:
                bear_tgt_on = False

        # ── 6. Prune broken FVGs ──────────────────────────────────────────
        active_bull = [f for f in active_bull if c >= f["bot"]]
        active_bear = [f for f in active_bear if c <= f["top"]]

        # ── 7. Record current stack state ─────────────────────────────────
        bull_stack_count[i] = len(active_bull)
        bear_stack_count[i] = len(active_bear)
        bull_stacking[i]    = float(len(active_bull) >= 2)
        bear_stacking[i]    = float(len(active_bear) >= 2)

        if active_bull:
            nb = active_bull[0]
            bull_near_top[i]  = nb["top"]
            bull_near_bot[i]  = nb["bot"]
            bull_near_size[i] = nb["top"] - nb["bot"]
            bull_near_mid[i]  = (nb["top"] + nb["bot"]) / 2
            dist_near_bull[i] = (c - nb["top"]) / c if c else np.nan
            if len(active_bull) >= 2:
                bull_next_top[i] = active_bull[1]["top"]
                bull_next_bot[i] = active_bull[1]["bot"]

        if active_bear:
            nb = active_bear[0]
            bear_near_top[i]  = nb["top"]
            bear_near_bot[i]  = nb["bot"]
            bear_near_size[i] = nb["top"] - nb["bot"]
            bear_near_mid[i]  = (nb["top"] + nb["bot"]) / 2
            dist_near_bear[i] = (nb["bot"] - c) / c if c else np.nan
            if len(active_bear) >= 2:
                bear_next_top[i] = active_bear[1]["top"]
                bear_next_bot[i] = active_bear[1]["bot"]

        # ── 8. Targeting state machine ────────────────────────────────────
        # Auto-expire if target not reached within window
        if bull_tgt_on and (i - bull_tgt_at) > target_expire_bars: bull_tgt_on = False
        if bear_tgt_on and (i - bear_tgt_at) > target_expire_bars: bear_tgt_on = False

        # Fire targeting signals — one-bar lag after inversion (bull_tgt_at < i)
        # so the signal is strictly forward-looking.
        if bull_tgt_on and bull_tgt_at < i:
            bull_targeting[i]  = 1.0
            bull_tgt_top_arr[i]= bull_tgt_top
            bull_tgt_bot_arr[i]= bull_tgt_bot
            dist_tgt_bull[i]   = (c - bull_tgt_top) / c if c else np.nan
            # Arrival: wick entered the zone and close held inside it
            if l <= bull_tgt_top and c >= bull_tgt_bot:
                bull_at_next[i] = 1.0
                bull_tgt_on     = False          # target reached — reset

        if bear_tgt_on and bear_tgt_at < i:
            bear_targeting[i]  = 1.0
            bear_tgt_top_arr[i]= bear_tgt_top
            bear_tgt_bot_arr[i]= bear_tgt_bot
            dist_tgt_bear[i]   = (bear_tgt_bot - c) / c if c else np.nan
            # Arrival: wick entered the zone and close held inside it
            if h >= bear_tgt_bot and c <= bear_tgt_top:
                bear_at_next[i] = 1.0
                bear_tgt_on     = False

    # ── Assign to DataFrame ───────────────────────────────────────────────────
    df["fvg_bull_stack_count"]    = bull_stack_count.astype("float32")
    df["fvg_bear_stack_count"]    = bear_stack_count.astype("float32")
    df["fvg_impulse_up_count"]    = impulse_up.astype("float32")
    df["fvg_impulse_dn_count"]    = impulse_dn.astype("float32")
    df["fvg_near_bull_top"]       = bull_near_top.astype("float32")
    df["fvg_near_bull_bot"]       = bull_near_bot.astype("float32")
    df["fvg_near_bull_size"]      = bull_near_size.astype("float32")
    df["fvg_near_bull_mid"]       = bull_near_mid.astype("float32")
    df["fvg_near_bear_top"]       = bear_near_top.astype("float32")
    df["fvg_near_bear_bot"]       = bear_near_bot.astype("float32")
    df["fvg_near_bear_size"]      = bear_near_size.astype("float32")
    df["fvg_near_bear_mid"]       = bear_near_mid.astype("float32")
    df["fvg_next_bull_top"]       = bull_next_top.astype("float32")
    df["fvg_next_bull_bot"]       = bull_next_bot.astype("float32")
    df["fvg_next_bear_top"]       = bear_next_top.astype("float32")
    df["fvg_next_bear_bot"]       = bear_next_bot.astype("float32")
    df["fvg_target_bull_top"]     = bull_tgt_top_arr.astype("float32")
    df["fvg_target_bull_bot"]     = bull_tgt_bot_arr.astype("float32")
    df["fvg_target_bear_top"]     = bear_tgt_top_arr.astype("float32")
    df["fvg_target_bear_bot"]     = bear_tgt_bot_arr.astype("float32")
    df["fvg_bull_inversion"]      = bull_inversion.astype("float32")
    df["fvg_bear_inversion"]      = bear_inversion.astype("float32")
    df["fvg_bull_stacking"]       = bull_stacking.astype("float32")
    df["fvg_bear_stacking"]       = bear_stacking.astype("float32")
    df["fvg_targeting_next_bull"] = bull_targeting.astype("float32")
    df["fvg_targeting_next_bear"] = bear_targeting.astype("float32")
    df["fvg_at_next_bull"]        = bull_at_next.astype("float32")
    df["fvg_at_next_bear"]        = bear_at_next.astype("float32")
    df["fvg_dist_to_near_bull"]   = dist_near_bull.astype("float32")
    df["fvg_dist_to_near_bear"]   = dist_near_bear.astype("float32")
    df["fvg_dist_to_target_bull"] = dist_tgt_bull.astype("float32")
    df["fvg_dist_to_target_bear"] = dist_tgt_bear.astype("float32")
    df["fvg_bull_inv_with_target"]= bull_inv_with_tgt.astype("float32")
    df["fvg_bear_inv_with_target"]= bear_inv_with_tgt.astype("float32")

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


def _detect_breaker_double_inverse(o, h, l, c, structure_lookback=20):
    """
    Single-pass, no-lookahead detector for ICT-style breaker-block
    "double inversions" on one series of OHLC arrays.

    Order block (OB)
    -----------------
    Bullish OB: the last bearish (red, close < open) candle immediately
    before a bar that closes above the recent rolling swing high — a
    bullish break of structure / impulse up. Its price range is the
    candle's real body: [min(open, close), max(open, close)].

    Bearish OB: the last bullish (green) candle immediately before a bar
    that closes below the recent rolling swing low — bearish break of
    structure. Same body-range definition.

    Breaker block
    -------------
    An OB becomes a "breaker" the first time price closes back through
    it against its original bias:
      - a bullish OB is broken (becomes a bearish breaker) when a later
        close < the OB's bottom.
      - a bearish OB is broken (becomes a bullish breaker) when a later
        close > the OB's top.

    Double inverse (what this function flags)
    -------------------------------------------
    A "double inverse" fires when a breaker block is itself broken back
    through in the OB's *original* direction — price reclaims the zone:
      - a bearish breaker (former bullish OB) double-inverses when a
        later close > the zone's top → bullish signal.
      - a bullish breaker (former bearish OB) double-inverses when a
        later close < the zone's bottom → bearish signal.

    Order/breaker blocks are never expired by age — once created they
    are remembered indefinitely and only dropped once they've fired
    their double-inverse signal, so tracking reaches back as far as the
    data provided allows.

    Returns
    -------
    (bull_double_inverse, bear_double_inverse) : two float arrays, same
    length as the input, 1.0 on the bar where the double-inverse closes,
    else 0.0.
    """
    n = len(c)
    bull_double_inverse = np.zeros(n)
    bear_double_inverse = np.zeros(n)

    if n < structure_lookback + 2:
        return bull_double_inverse, bear_double_inverse

    prior_high = pd.Series(h).shift(1).rolling(structure_lookback, min_periods=1).max().values
    prior_low  = pd.Series(l).shift(1).rolling(structure_lookback, min_periods=1).min().values

    bull_obs: list = []   # each: {"top", "bot", "created", "state"}; state: "ob" -> "breaker" -> consumed
    bear_obs: list = []

    for i in range(n):
        # ---- 1. New order block creation on structure break ----
        if not np.isnan(prior_high[i]) and c[i] > prior_high[i]:
            j = i - 1
            while j >= 0 and c[j] >= o[j]:      # scan back for nearest bearish candle
                j -= 1
            if j >= 0:
                bull_obs.append({
                    "top": max(o[j], c[j]), "bot": min(o[j], c[j]),
                    "created": j, "state": "ob",
                })

        if not np.isnan(prior_low[i]) and c[i] < prior_low[i]:
            j = i - 1
            while j >= 0 and c[j] <= o[j]:      # scan back for nearest bullish candle
                j -= 1
            if j >= 0:
                bear_obs.append({
                    "top": max(o[j], c[j]), "bot": min(o[j], c[j]),
                    "created": j, "state": "ob",
                })

        # ---- 2. Advance state machine for existing bullish OBs ----
        still_bull = []
        for ob in bull_obs:
            consumed = False
            if i > ob["created"]:
                if ob["state"] == "ob" and c[i] < ob["bot"]:
                    ob["state"] = "breaker"
                elif ob["state"] == "breaker" and c[i] > ob["top"]:
                    bull_double_inverse[i] = 1.0
                    consumed = True
            if not consumed:
                still_bull.append(ob)
        bull_obs = still_bull

        # ---- 3. Advance state machine for existing bearish OBs ----
        still_bear = []
        for ob in bear_obs:
            consumed = False
            if i > ob["created"]:
                if ob["state"] == "ob" and c[i] > ob["top"]:
                    ob["state"] = "breaker"
                elif ob["state"] == "breaker" and c[i] < ob["bot"]:
                    bear_double_inverse[i] = 1.0
                    consumed = True
            if not consumed:
                still_bear.append(ob)
        bear_obs = still_bear

    return bull_double_inverse, bear_double_inverse


def _add_breaker_blocks_single_tf(raw_df, structure_lookback=20):
    """Run the breaker-block double-inverse detector on one timeframe's OHLC data."""
    o = raw_df["Open"].astype(float).values
    h = raw_df["High"].astype(float).values
    l = raw_df["Low"].astype(float).values
    c = raw_df["Close"].astype(float).values
    bull_di, bear_di = _detect_breaker_double_inverse(o, h, l, c, structure_lookback)
    return pd.DataFrame(
        {"bull_double_inverse": bull_di, "bear_double_inverse": bear_di},
        index=raw_df.index,
    )


def _align_signal_to_primary(sig_df, primary_index, tf_minutes):
    """
    Align a lower-timeframe signal onto the primary dataframe's index with
    no lookahead. If the source timeframe is finer than the primary bar
    spacing, first take a rolling MAX over the number of finer bars that
    fit inside one primary bar, so a fast intrabar double-inverse isn't
    missed between two primary bars. Then carry the value forward with a
    backward (point-in-time) as-of merge.
    """
    sig_df = sig_df.copy()
    sig_df.index.name = "ts"

    primary_freq = pd.Series(primary_index).diff().median()
    tf_delta = pd.Timedelta(minutes=tf_minutes)
    if pd.notna(primary_freq) and tf_delta < primary_freq:
        window_bars = max(1, int(primary_freq / tf_delta))
        sig_df = sig_df.rolling(window_bars, min_periods=1).max()

    sig_df = sig_df.sort_index()
    left = pd.DataFrame({"ts": pd.DatetimeIndex(primary_index)}).sort_values("ts")
    merged = pd.merge_asof(left, sig_df.reset_index(), on="ts", direction="backward")
    merged = merged.set_index("ts").reindex(pd.DatetimeIndex(primary_index))
    return merged[["bull_double_inverse", "bear_double_inverse"]].fillna(0.0)


def _add_breaker_block_features(df, ticker=None, structure_lookback=20):
    """
    Multi-timeframe ICT breaker-block "double inverse" signals.

    See _detect_breaker_double_inverse for the exact order-block /
    breaker / double-inverse definitions. This wrapper independently
    downloads 1m, 5m, and 15m history for `ticker`, runs the detector on
    each timeframe, and aligns the resulting signals onto df's index
    (point-in-time, no lookahead).

    Order/breaker blocks are tracked with unlimited memory — they are
    never dropped for being "too old," only once they've fired their
    double-inverse signal — so detection reaches back as far as the
    downloaded history on each timeframe allows. Yahoo Finance itself
    caps how much history it will serve regardless of what's requested:
      • 1m bars       → roughly the last 7 days only
      • 5m/15m bars   → roughly the last 60 days only
    so "as far back as possible" is bounded by those Yahoo limits, not
    by anything in this function.

    Requires `ticker` (e.g. "AAPL") since these are downloaded
    independently of whatever interval df itself is already on. If no
    ticker is supplied, all columns below are filled with 0.0.

    Columns added (tf in {"1m", "5m", "15m"}):
      breaker_bull_double_inverse_{tf}   # bullish breaker reclaimed upward
      breaker_bear_double_inverse_{tf}   # bearish breaker reclaimed downward
    Plus, True if ANY timeframe fired on that bar:
      breaker_bull_double_inverse_any
      breaker_bear_double_inverse_any
    """
    timeframes = {"1m": ("7d", 1), "5m": ("60d", 5), "15m": ("60d", 15)}

    if ticker is None:
        print("[breaker_blocks] No ticker supplied — skipping multi-timeframe "
              "breaker-block features (filled with 0).")
        for tf in timeframes:
            df[f"breaker_bull_double_inverse_{tf}"] = 0.0
            df[f"breaker_bear_double_inverse_{tf}"] = 0.0
        df["breaker_bull_double_inverse_any"] = 0.0
        df["breaker_bear_double_inverse_any"] = 0.0
        return df

    for tf, (max_period, tf_minutes) in timeframes.items():
        try:
            raw = yf.download(ticker, period=max_period, interval=tf,
                               progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna(subset=["Open", "High", "Low", "Close"])
            if raw.empty:
                raise ValueError("empty download")

            sig = _add_breaker_blocks_single_tf(raw, structure_lookback=structure_lookback)
            aligned = _align_signal_to_primary(sig, df.index, tf_minutes)
            bull_col = aligned["bull_double_inverse"]
            bear_col = aligned["bear_double_inverse"]
        except Exception as e:
            print(f"[breaker_blocks] {tf} download/processing failed: {e}. Filling with 0.")
            bull_col = pd.Series(0.0, index=df.index)
            bear_col = pd.Series(0.0, index=df.index)

        df[f"breaker_bull_double_inverse_{tf}"] = bull_col.astype("float32").values
        df[f"breaker_bear_double_inverse_{tf}"] = bear_col.astype("float32").values

    bull_cols = [f"breaker_bull_double_inverse_{tf}" for tf in timeframes]
    bear_cols = [f"breaker_bear_double_inverse_{tf}" for tf in timeframes]
    df["breaker_bull_double_inverse_any"] = df[bull_cols].max(axis=1)
    df["breaker_bear_double_inverse_any"] = df[bear_cols].max(axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, period: str = "30d", interval: str = "5m", ticker: str = None) -> pd.DataFrame:
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
    ticker : str, optional
        Symbol string (e.g. "AAPL") used to independently download 1m/5m/15m
        history for the multi-timeframe breaker-block features. If omitted,
        those columns are filled with 0.

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
    df = _add_fvg_stack_features(df)
    df = _add_index_divergence(df, period=period, interval=interval)
    df = _add_breaker_block_features(df, ticker=ticker)

    # Safety net: force every boolean-signal column to float32
    bool_signal_cols = [c for c in df.columns if c in FEATURES and c != "Close"]
    for col in bool_signal_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df




# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────
# Every column produced by the full pipeline is listed here, grouped by
# the function / step that generates it.  This acts as both documentation
# and the column-selection spec for the output CSV.
# ─────────────────────────────────────────────────────────────────────────────

_NEWS_COLS = [
    "news_sentiment_score",       # continuous EW-mean score in [-1, 1]
    "news_article_count",         # number of articles in the rolling window
    "news_bull",                  # score > +0.15
    "news_bear",                  # score < -0.15
    "news_sentiment_strong_bull", # score > +0.40
    "news_sentiment_strong_bear", # score < -0.40
]


FEATURES = [

    # ── continuous price ───────────────────────────────────────────────────
    "Close",

    # # ── liquidity sweeps  (_add_liquidity_sweeps) ──────────────────────────
    # "prev_day_high",       # previous session high (price level)
    # "prev_day_low",        # previous session low  (price level)
    # "high_wick_sweep",     # wick pierced prev high but closed below it
    # "low_wick_sweep",      # wick pierced prev low  but closed above it

    # # ── candle direction  (_add_candle_features) ───────────────────────────
    # "isGreen",             # close > open
    # "isHigh",              # green candle AND higher close than prior bar
    # "isLow",               # red candle AND lower close than prior bar

    # # ── structural trend  (_add_structural_trend) ──────────────────────────
    # # drop_first=True drops "trend_downtrend" (alphabetically first).
    # # Encoding: downtrend → (trend_ranging=0, trend_uptrend=0)
    # "trend_ranging",
    # "trend_uptrend",

    # # ── ADX trend strength  (_add_adx) ────────────────────────────────────
    # "ADX",
    # "DMP",                    # +DI
    # "DMN",                    # -DI
    # "is_trending",            # ADX > 25
    # "adx_uptrend",            # ADX > 25 AND +DI > -DI
    # "adx_downtrend",          # ADX > 25 AND -DI > +DI
    # "DI_cross_up",            # +DI just crossed above -DI this bar
    # "DI_cross_down",          # -DI just crossed above +DI this bar
    # "ADX_slope",              # 3-bar change in ADX
    # "trend_strengthening",    # ADX slope > 0
    # "trend_weakening",        # ADX slope < 0
    # "strong_up",              # adx_uptrend AND strengthening
    # "fading_up",              # adx_uptrend AND weakening
    # "strong_down",            # adx_downtrend AND strengthening
    # "fading_down",            # adx_downtrend AND weakening

    # # ── MACD + Linear Regression Channel  (_add_macd_lr) ──────────────────
    # "macd_line",
    # "macd_signal",
    # "macd_hist",
    # "lr_upper",               # LR midline + 4σ
    # "lr_lower",               # LR midline − 4σ
    # "near_upper_band",        # close ≥ midline + 0.8×dev (strong bullish)
    # "near_lower_band",        # close ≤ midline − 0.8×dev (strong bearish)
    # "touches_upper",          # close ≥ upper band (stretched)
    # "touches_lower",          # close ≤ lower band (compressed)
    # "broke_above",            # just crossed above upper band
    # "broke_below",            # just crossed below lower band
    # "bands_widening",         # band width > width 3 bars ago
    # "bands_narrowing",        # band width < width 3 bars ago
    # "macd_bullish_entry",     # MACD cross up while near lower band
    # "macd_bearish_entry",     # MACD cross down while near upper band
    # "rsi",
    # "rsi_overbought",         # RSI > 70
    # "rsi_oversold",           # RSI < 30

    # # ── mean reversion  (_add_mean_reversion) ─────────────────────────────
    # "mr_below_sma20",
    # "mr_above_sma20",
    # "mr_bb_below_lower",      # below lower Bollinger Band (2σ)
    # "mr_bb_above_upper",      # above upper Bollinger Band (2σ)
    # "mr_rsi_oversold",
    # "mr_rsi_overbought",
    # "mr_z_score_low",         # Z-score < −1.5
    # "mr_z_score_high",        # Z-score >  1.5
    # "mr_below_vwap",
    # "mr_above_vwap",

    # # ── momentum  (_add_momentum) ──────────────────────────────────────────
    # "mo_roc_positive_20",     # 20-bar rate-of-change > 0
    # "mo_roc_negative_20",
    # "mo_golden_cross",        # SMA50 just crossed above SMA200
    # "mo_death_cross",
    # "mo_macd_cross_up",
    # "mo_macd_cross_down",
    # "mo_adx_trending",        # ADX > 25
    # "mo_breakout_high20",     # close > 20-bar rolling high (no lookahead)
    # "mo_breakdown_low20",
    # "mo_volume_surge",        # volume > 2× 20-bar average
    # "mo_consecutive_up3",     # 3+ consecutive green bars
    # "mo_consecutive_down3",
    # "mo_combo_long",          # ≥2 of 5 bullish momentum signals firing
    # "mo_combo_short",

    # # ── ICT liquidity sweep → FVG setups  (_add_sweep_fvg_setups) ─────────
    # "fvg_bull_top",           # top of active bullish FVG (NaN = no active zone)
    # "fvg_bull_bot",
    # "fvg_bear_top",
    # "fvg_bear_bot",
    # "in_bull_fvg",            # price is currently inside a bullish FVG
    # "in_bear_fvg",
    # "recent_low_sweep",       # low sweep within last sweep_lookback bars
    # "recent_high_sweep",
    # "setup_bull_sweep_fvg",   # recent low sweep + currently in bull FVG
    # "setup_bear_sweep_fvg",
    # "setup_bull_confirmed",   # sweep+FVG setup confirmed by green candle
    # "setup_bear_confirmed",

    # # ── FVG Stack + Sequential Inversion Targeting  (_add_fvg_stack_features) ─
    # "fvg_bull_stack_count",      # # active bull FVGs still in stack
    # "fvg_bear_stack_count",      # # active bear FVGs still in stack
    # "fvg_impulse_up_count",      # bull FVGs created in last stack_lookback bars (impulse strength)
    # "fvg_impulse_dn_count",      # bear FVGs created in last stack_lookback bars

    # "fvg_near_bull_top",         # price: top of nearest (highest) active bull FVG
    # "fvg_near_bull_bot",         # price: bottom of nearest active bull FVG
    # "fvg_near_bull_size",        # gap width of nearest bull FVG
    # "fvg_near_bull_mid",         # midpoint of nearest bull FVG
    # "fvg_near_bear_top",         # price: top of nearest (lowest) active bear FVG
    # "fvg_near_bear_bot",         # price: bottom of nearest active bear FVG
    # "fvg_near_bear_size",        # gap width of nearest bear FVG
    # "fvg_near_bear_mid",         # midpoint of nearest bear FVG

    # "fvg_next_bull_top",         # price: 2nd bull FVG in the stack (below nearest)
    # "fvg_next_bull_bot",
    # "fvg_next_bear_top",         # price: 2nd bear FVG in the stack (above nearest)
    # "fvg_next_bear_bot",

    # "fvg_target_bull_top",       # price: top of the pinned bull target FVG (NaN if inactive)
    # "fvg_target_bull_bot",
    # "fvg_target_bear_top",       # price: top of the pinned bear target FVG
    # "fvg_target_bear_bot",

    # "fvg_bull_inversion",        # 1 when nearest bull FVG inversed (close < bot)
    # "fvg_bear_inversion",        # 1 when nearest bear FVG inversed (close > top)
    # "fvg_bull_stacking",         # 1 if ≥2 active bull FVGs (impulse regime detected)
    # "fvg_bear_stacking",         # 1 if ≥2 active bear FVGs

    # "fvg_targeting_next_bull",   # 1 while in post-inversion targeting state (bull)
    # "fvg_targeting_next_bear",   # 1 while in post-inversion targeting state (bear)
    # "fvg_at_next_bull",          # 1 when price first arrives at the next bull FVG target
    # "fvg_at_next_bear",          # 1 when price first arrives at the next bear FVG target

    # "fvg_dist_to_near_bull",     # (close − near_top) / close; neg = price inside or below zone
    # "fvg_dist_to_near_bear",     # (near_bot − close) / close; neg = price inside or above zone
    # "fvg_dist_to_target_bull",   # normalized distance to active bull target (NaN if inactive)
    # "fvg_dist_to_target_bear",   # normalized distance to active bear target

    # "fvg_bull_inv_with_target",  # inversion fired AND next-target FVG exists — the core setup
    # "fvg_bear_inv_with_target",

    # # ── SPY / QQQ index divergence  (_add_index_divergence) ───────────────
    # "spy_ret",                # SPY bar-over-bar return
    # "qqq_ret",                # QQQ bar-over-bar return
    # "idx_div_spy_up_qqq_down",
    # "idx_div_spy_down_qqq_up",
    # "idx_div_any",
    # "idx_div_rolling3",       # divergence persisted in any of last 3 bars
    # "idx_corr_20",            # 20-bar rolling SPY/QQQ correlation
    # "idx_corr_breakdown",     # rolling correlation < 0.5 (regime decoupling)

    # ── ICT breaker blocks — double inverse  (_add_breaker_block_features) ─
    "breaker_bull_double_inverse_1m",   # bullish reclaim of a bearish breaker, 1m
    "breaker_bear_double_inverse_1m",   # bearish reclaim of a bullish breaker, 1m
    "breaker_bull_double_inverse_5m",
    "breaker_bear_double_inverse_5m",
    "breaker_bull_double_inverse_15m",
    "breaker_bear_double_inverse_15m",
    "breaker_bull_double_inverse_any",  # fired on ANY of the 3 timeframes
    "breaker_bear_double_inverse_any",

    # ── FinBERT / Finnhub news sentiment  (step 3 — optional) ─────────────
    # Only present when USE_NEWS_SENTIMENT = True.
    # *_NEWS_COLS,
]