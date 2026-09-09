"""
dataset_builder.py
========================
Downloads OHLCV data, runs the full feature engineering pipeline, optionally
adds FinBERT news sentiment features, then labels each bar BUY / SELL / HOLD
based on forward price returns.

Pipeline order
--------------
  1. Download OHLCV (yfinance)
  2. Build technical features (feature_engineering.build_features)
  3. Add news sentiment features (news_sentiment.SentimentPipeline)   ← optional
  4. Label each bar BUY / SELL / HOLD
  5. Clean, save, summarise

Labelling methods (configure via LABEL_METHOD):
  "fixed_pct"   – BUY if fwd_ret >  +PCT_THRESHOLD
                   SELL if fwd_ret < -PCT_THRESHOLD, else HOLD
  "atr_mult"    – threshold = ATR_MULT × ATR(14) expressed as % of price
  "rolling_std" – threshold = STD_MULT × 20-bar rolling σ of returns

Target integer codes
  2 = BUY  |  1 = HOLD  |  0 = SELL

Usage
-----
  python build_labeled_dataset.py

Output
------
  labeled_dataset.csv  (all features + target, NaN-free, ML-ready)
"""

import os
import re
import numpy as np
import pandas as pd
import yfinance as yf

from feature_engineering import build_features, FEATURES

import mplfinance as mpf
import pandas as pd

def plot_target_signals(raw_ohlcv: pd.DataFrame, labeled: pd.DataFrame,
                         lookback_bars: int = 300,
                         title: str = "Labeled BUY/SELL Targets",
                         buy_code: int = 2, sell_code: int = 0):
    """
    Plot a candlestick chart with green up-arrows on bars labeled BUY and
    red down-arrows on bars labeled SELL (from the `target` column produced
    by build_labeled_dataset() / label_signals()).

    Parameters
    ----------
    raw_ohlcv : pd.DataFrame
        Original OHLCV data (Open/High/Low/Close/Volume, DatetimeIndex).
    labeled : pd.DataFrame
        Output of build_labeled_dataset() — must contain a 'target' column
        sharing the same index (or a subset of it) as raw_ohlcv.
    lookback_bars : int
        Only plot the most recent N bars. Set to None to plot everything.
    buy_code, sell_code : int
        Target integer codes — defaults match dataset_builder.py (BUY=2, SELL=0).
        HOLD (1) bars get no marker.
    """
    df = raw_ohlcv.copy()

    # labeled may have fewer rows than raw (tail dropped for unlabelable bars),
    # so align by reindexing and leave HOLD/missing as NaN → no marker drawn.
    target = labeled["target"].reindex(df.index)

    if lookback_bars is not None:
        df = df.iloc[-lookback_bars:]
        target = target.iloc[-lookback_bars:]

    buy_marker  = (df["Low"]  * 0.999).where(target == buy_code)
    sell_marker = (df["High"] * 1.001).where(target == sell_code)

    addplots = []
    if buy_marker.notna().any():
        addplots.append(
            mpf.make_addplot(buy_marker, type="scatter", markersize=100,
                              marker="^", color="green")
        )
    if sell_marker.notna().any():
        addplots.append(
            mpf.make_addplot(sell_marker, type="scatter", markersize=100,
                              marker="v", color="red")
        )

    mpf.plot(
        df,
        type="candle",
        style="charles",
        title=title,
        addplot=addplots,
        volume=True,
        figsize=(16, 8),
        tight_layout=True,
    )



# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ←  everything you need to edit lives here
# ─────────────────────────────────────────────────────────────────────────────

TICKER   = "BTC"
PERIOD   = "60d"    # yfinance download window
INTERVAL = "5m"     # bar frequency; must match build_features() expectation

FORWARD_BARS  = 8           # look-ahead bars for labelling (18 × 5 m = 90 min)

LABEL_METHOD  = "atr_mult"  # "fixed_pct" | "atr_mult" | "rolling_std"
PCT_THRESHOLD = 0.05       # ±0.30 %  — only used when LABEL_METHOD = "fixed_pct"
ATR_MULT      = 4.0         # ATR multiplier — only used when LABEL_METHOD = "atr_mult"
STD_MULT      = 1.0         # std multiplier — only used when LABEL_METHOD = "rolling_std"

# ── News sentiment ────────────────────────────────────────────────────────────
# Set USE_NEWS_SENTIMENT = False to skip the Finnhub + FinBERT step entirely
# (useful when you haven't set up the API key yet or want a faster run).
USE_NEWS_SENTIMENT = True

# Finnhub API key.  Leave as "" to read from the FINNHUB_API_KEY env var.
FINNHUB_API_KEY = ""

# Extra calendar days added to the news fetch window beyond what PERIOD covers.
# e.g. PERIOD="60d" → fetch 65 days of news so no headline near the start
# of the price window gets missed.
SENTIMENT_DAYS_BUFFER = 5

OUTPUT_PATH           = "labeled_dataset.csv"
OUTPUT_PATH_UNLABELED = "unlabeled_dataset.csv"

# Target codes
BUY  = 2
HOLD = 1
SELL = 0

LABEL_NAMES = {BUY: "BUY", HOLD: "HOLD", SELL: "SELL"}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────
# The technical feature catalogue is NOT duplicated here — it's imported
# directly from feature_engineering.FEATURES, which is the single source of
# truth for "what columns does the technical pipeline produce, and which of
# them do we want in the output." Whatever is active (uncommented) in that
# list there is exactly what shows up here — nothing to keep in sync by hand.
#
# News sentiment columns are a separate concern: feature_engineering.py has
# no knowledge of them (they're computed here, in _add_sentiment(), not in
# build_features()), so they're tracked independently and combined with
# FEATURES below to form the full catalogue.
# ─────────────────────────────────────────────────────────────────────────────

# Columns that only exist when USE_NEWS_SENTIMENT = True.
# Isolated so the "missing columns" warning can tell the difference between
# "this is a real bug" and "sentiment was just turned off".
_NEWS_COLS = [
    "news_sentiment_score",       # continuous EW-mean score in [-1, 1]
    "news_article_count",         # number of articles in the rolling window
    "news_bull",                  # score > +0.15
    "news_bear",                  # score < -0.15
    "news_sentiment_strong_bull", # score > +0.40
    "news_sentiment_strong_bear", # score < -0.40
]

# FVG price-level columns are legitimately NaN when no FVG is active.
# NaN there means "not inside any fair-value gap zone" → filled with 0.
_FVG_ZONE_COLS = [
    "fvg_bull_top",
    "fvg_bull_bot",
    "fvg_bear_top",
    "fvg_bear_bot",
]

# Full catalogue = whatever feature_engineering.py currently produces/selects
# (FEATURES) + the news columns this file adds on its own — but only when
# USE_NEWS_SENTIMENT is actually on. Previously _NEWS_COLS was appended
# unconditionally, which meant news booleans (news_bull, news_bear, etc.)
# leaked into the training set even when FEATURES had nothing but a few
# technical columns uncommented — silently defeating the point of using
# FEATURES to control what the model trains on.
ALL_FEATURE_COLS = FEATURES + (_NEWS_COLS if USE_NEWS_SENTIMENT else [])


def _is_binary_col(series: pd.Series) -> bool:
    """
    True if every non-null value in `series` is 0 or 1 (i.e. a boolean-style
    signal column rather than a continuous indicator).

    This replaces a hand-maintained "which columns are boolean" list: instead
    of keeping a second catalogue that has to be updated every time
    feature_engineering.py adds/removes a column, the boolean subset of
    FEATURES is detected directly from the data. Add a new signal column to
    feature_engineering.FEATURES and it's automatically picked up here with
    zero changes needed in this file.
    """
    vals = pd.to_numeric(series, errors="coerce").dropna().unique()
    return set(np.unique(vals)).issubset({0.0, 1.0})


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _period_to_days(period: str) -> int:
    """
    Convert a yfinance period string to an approximate number of calendar days.

    Handles the formats yfinance accepts:
        "60d"  → 60
        "2mo"  → 60   (1 mo ≈ 30 days)
        "1y"   → 365
        "ytd"  → 365  (conservative upper bound)
        "max"  → 730  (safe upper bound; Finnhub free tier caps anyway)
    """
    period = period.strip().lower()
    match  = re.fullmatch(r"(\d+)(d|mo|y)", period)
    if match:
        n, unit = int(match.group(1)), match.group(2)
        return n * {"d": 1, "mo": 30, "y": 365}[unit]
    return 365   # fallback for "ytd", "max", unrecognised strings

def dedupe_consecutive_signals(signal: pd.Series) -> pd.Series:
    """
    Collapse consecutive runs of the same BUY/SELL label to just the last
    bar in each run. Overlapping forward-return windows mean several bars
    in a row often cross the threshold right before the same big move —
    this keeps only the final trigger (closest to the move) and turns the
    rest of the run back to HOLD.
    """
    out = signal.copy()
    is_signal = out.isin([BUY, SELL])
    # True where this bar's label differs from the NEXT bar's label
    # (i.e. this bar is the last one in its run)
    end_of_run = is_signal & (out != out.shift(-1))
    out.loc[is_signal & ~end_of_run] = HOLD
    return out


def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Exponential ATR (Wilder-style EMA on True Range)."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()


def _build_threshold(df: pd.DataFrame) -> pd.Series:
    """
    Per-bar threshold Series (expressed as a return fraction, always positive).
    Method and constants are read from module-level config.
    """
    close = df["Close"].squeeze()

    if LABEL_METHOD == "fixed_pct":
        return pd.Series(PCT_THRESHOLD, index=df.index)

    elif LABEL_METHOD == "atr_mult":
        # Volatility-adaptive: wider in volatile regimes, tighter when calm.
        atr = _compute_atr(df)
        return (atr / close) * ATR_MULT

    elif LABEL_METHOD == "rolling_std":
        # Sigma-based: threshold scales with recent return dispersion.
        returns = close.pct_change()
        return returns.rolling(20, min_periods=5).std() * STD_MULT

    else:
        raise ValueError(
            f"Unknown LABEL_METHOD='{LABEL_METHOD}'. "
            "Choose 'fixed_pct', 'atr_mult', or 'rolling_std'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL LABELLING
# ─────────────────────────────────────────────────────────────────────────────

def label_signals(df: pd.DataFrame) -> pd.Series:
    """
    Assign BUY / HOLD / SELL to every bar using the forward return over the
    next FORWARD_BARS bars:

        fwd_return = (close[t + FORWARD_BARS] / close[t]) − 1

    Decision rules (mutually exclusive by construction)
    ---------------------------------------------------
    fwd_return >  threshold  →  BUY  (2)
    fwd_return < −threshold  →  SELL (0)
    otherwise                →  HOLD (1)

    The last FORWARD_BARS rows receive NaN — no future close exists.

    Returns
    -------
    pd.Series (float): 2.0 / 1.0 / 0.0 / NaN for the unlabelable tail.
    """
    close      = df["Close"].squeeze()
    fwd_return = close.shift(-FORWARD_BARS) / close - 1
    threshold  = _build_threshold(df)

    signal = pd.Series(np.nan, index=df.index, dtype=float)
    valid  = fwd_return.notna() & threshold.notna()

    signal.loc[valid & (fwd_return >  threshold)] = BUY
    signal.loc[valid & (fwd_return < -threshold)] = SELL
    signal.loc[
        valid &
        (fwd_return >= -threshold) &
        (fwd_return <=  threshold)
    ] = HOLD

    return signal


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT STEP  (isolated so failures here never crash the whole pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _add_sentiment(df: pd.DataFrame, ticker: str, period: str) -> pd.DataFrame:
    """
    Attempt to run the FinBERT / Finnhub sentiment pipeline and merge the
    resulting columns into df.  Returns df unchanged if anything goes wrong
    (import error, missing API key, network failure, etc.).

    Why isolated?
    -------------
    The sentiment step has three external failure modes that the rest of the
    pipeline doesn't: network access to Finnhub, loading a large PyTorch model,
    and a Finnhub API key.  Wrapping it here means a misconfigured key or a
    Finnhub outage doesn't prevent you from building the technical feature set.
    """
    try:
        from news_sentiment import SentimentPipeline
    except ImportError as e:
        print(f"  ⚠  Could not import news_sentiment: {e}")
        print("     Install: pip install transformers torch finnhub-python")
        print("     Skipping sentiment features.")
        return df

    api_key   = FINNHUB_API_KEY or os.environ.get("FINNHUB_API_KEY", "")
    days_back = _period_to_days(period) + SENTIMENT_DAYS_BUFFER

    if not api_key:
        print("  ⚠  No Finnhub API key found.")
        print("     Set FINNHUB_API_KEY env var or set FINNHUB_API_KEY in config.")
        print("     Skipping sentiment features.")
        return df

    try:
        pipe = SentimentPipeline(
            ticker    = ticker,
            api_key   = api_key,
            days_back = days_back,
        )
        df = pipe.add_sentiment_features(df)
        print(f"  ✓  Sentiment features added "
              f"({days_back} days of headlines, {len(_NEWS_COLS)} columns).")
    except Exception as e:
        print(f"  ⚠  Sentiment pipeline failed: {e}")
        print("     Skipping sentiment features. Technical features are unaffected.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PIPELINE CORE  (private)
# ─────────────────────────────────────────────────────────────────────────────

def _build_enriched(
    ticker:   str,
    period:   str,
    interval: str,
    step_prefix: str = "",   # e.g. "[1/4]" vs "[1/5]" depending on caller
) -> tuple[pd.DataFrame, list[str]]:
    """
    Shared steps used by both public functions:
        [1] Download raw OHLCV from yfinance
        [2] Build all technical indicator features
        [3] Optionally add FinBERT / Finnhub news sentiment features

    Returns
    -------
    enriched  : pd.DataFrame  — all OHLCV + feature columns, no target yet
    feat_cols : list[str]     — ordered list of feature columns that exist
                                in enriched (subset of ALL_FEATURE_COLS)
    """
    def step(n, total, msg):
        print(f"{step_prefix}[{n}/{total}] {msg}")

    total = 3

    # ── [1] Download ─────────────────────────────────────────────────────────
    step(1, total, f"Downloading {ticker}  period={period}  interval={interval} …")
    raw = yf.download(
        ticker, period=period, interval=interval,
        progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    print(f"      {len(raw):,} bars downloaded.")

    # ── [2] Technical features ────────────────────────────────────────────────
    step(2, total, "Building technical features …")
    print("      (Note: _add_structural_trend has an O(n·lookback) Python loop;")
    print("       expect ~20–40 s for 60 days of 5-min data.)")
    enriched = build_features(raw, period=period, interval=interval, ticker=ticker)
    print(f"      Done — {enriched.shape[1]} columns after technical features.")
    print(f"      Lorentzian cols present: {[c for c in enriched.columns if 'lorentzian' in c.lower()]}")   # ADD THIS

    # ── [3] News sentiment ────────────────────────────────────────────────────
    if USE_NEWS_SENTIMENT:
        step(3, total, "Adding FinBERT news sentiment features …")
        enriched = _add_sentiment(enriched, ticker=ticker, period=period)
        n_news = sum(c in enriched.columns for c in _NEWS_COLS)
        print(f"      {n_news}/{len(_NEWS_COLS)} sentiment columns present "
              f"({enriched.shape[1]} total columns).")
    else:
        step(3, total, "News sentiment skipped (USE_NEWS_SENTIMENT = False).")

    # Resolve which feature columns actually exist
    feat_cols = [c for c in ALL_FEATURE_COLS if c in enriched.columns]

    # Warn about genuinely unexpected missing columns
    optional = set(_NEWS_COLS if not USE_NEWS_SENTIMENT else [])
    missing  = [
        c for c in ALL_FEATURE_COLS
        if c not in enriched.columns and c not in optional
    ]
    if missing:
        print(f"  ⚠  {len(missing)} expected column(s) not found: {missing}")

    return enriched, feat_cols


def _clean_features(
    df:        pd.DataFrame,
    feat_cols: list[str],
) -> pd.DataFrame:
    """
    Shared NaN-handling applied to the feature matrix by both public functions.

    FVG zone columns  → filled with 0  (NaN = no active zone, not missing data)
    Sentiment columns → filled with 0  (NaN = no news yet = neutral)
    Indicator warm-up → forward-filled, then zero-filled
    """
    fvg_fill  = [c for c in _FVG_ZONE_COLS if c in df.columns]
    news_fill = [c for c in _NEWS_COLS      if c in df.columns]

    if fvg_fill:
        df[fvg_fill]  = df[fvg_fill].fillna(0)
    if news_fill:
        df[news_fill] = df[news_fill].fillna(0)

    df[feat_cols] = df[feat_cols].ffill().fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION 1  —  LABELED DATASET  (supervised learning)
# ─────────────────────────────────────────────────────────────────────────────

def build_labeled_dataset(
    ticker:   str = TICKER,
    period:   str = PERIOD,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """
    Build a fully labeled, ML-ready dataset for supervised learning.

    Columns kept
    ------------
    Close        — the only continuous feature; price context for the model
    bool_cols    — every column from feature_engineering.FEATURES (+ news
                   cols) whose values are all 0/1 in the actual data (see
                   _is_binary_col) — detected live, not hand-listed
    target       — int  2=BUY  1=HOLD  0=SELL

    Continuous indicators (ADX, RSI, MACD values, price levels, raw returns)
    are intentionally excluded so the model learns from discrete signals only.

    Pipeline
    --------
    [1] Download OHLCV
    [2] Build all technical features
    [3] Add news sentiment (optional)
    [4] Label every bar BUY / SELL / HOLD from forward price returns
    [5] Select Close + boolean cols, clean NaN, drop unlabelable tail

    NaN handling
    ------------
    tail rows   Dropped — forward return undefined for last FORWARD_BARS bars.
    FVG zones   → 0  (no active zone)
    sentiment   → 0  (no news = neutral)
    warm-up     → ffill then 0-fill
    """
    bar_num  = int("".join(filter(str.isdigit, interval)))
    bar_unit = interval[-1]

    print(f"\n{'═' * 55}")
    print(f"  Building LABELED dataset  ({ticker})")
    print(f"{'═' * 55}")

    # ── Shared steps [1–3] ───────────────────────────────────────────────────
    # _build_enriched generates ALL features; we narrow down to Close + booleans
    # in step [5] below rather than restricting what gets computed, so that
    # label_signals() still has access to Close, High, Low for thresholds.
    enriched, feat_cols = _build_enriched(ticker, period, interval)

    # ── [4] Label ─────────────────────────────────────────────────────────────
    print(
        f"[4/5] Labelling with method='{LABEL_METHOD}', "
        f"forward_bars={FORWARD_BARS} "
        f"(≈ {FORWARD_BARS * bar_num} {bar_unit} look-ahead) …"
    )
    enriched["target"] = label_signals(enriched)
    # enriched["target"] = dedupe_consecutive_signals(enriched["target"])

    # ── [5] Finalise ──────────────────────────────────────────────────────────
    print("[5/5] Selecting Close + features, cleaning and finalising …")

    candidates = [
    c for c in feat_cols
    if c != "Close" and c in enriched.columns and _is_binary_col(enriched[c])
]
    keep       = list(dict.fromkeys(["Close"] + candidates + ["target"]))
    dataset    = enriched[keep].copy()

    n_before = len(dataset)
    dataset  = dataset.dropna(subset=["target"])
    n_after  = len(dataset)
    print(f"  Dropped {n_before - n_after:,} tail rows. {n_after:,} rows remain.")

    dataset = _clean_features(dataset, candidates)

    # Safe to cast now that all NaN rows are gone
    dataset["target"] = dataset["target"].astype(int)

    # ── Summary ───────────────────────────────────────────────────────────────
    news_bool_present = [c for c in ["news_bull", "news_bear",
                                     "news_sentiment_strong_bull",
                                     "news_sentiment_strong_bear"]
                         if c in dataset.columns]
    total = len(dataset)
    print(f"\n{'─' * 55}")
    print(f"  Labeled dataset summary")
    print(f"{'─' * 55}")
    print(f"  Ticker          : {ticker}")
    print(f"  Period/Interval : {period} / {interval}")
    print(f"  Label method    : {LABEL_METHOD}", end="")
    if   LABEL_METHOD == "fixed_pct":   print(f"  (±{PCT_THRESHOLD:.3%})")
    elif LABEL_METHOD == "atr_mult":    print(f"  ({ATR_MULT}× ATR14 / Close)")
    elif LABEL_METHOD == "rolling_std": print(f"  ({STD_MULT}× 20-bar σ)")
    print(f"  Forward bars    : {FORWARD_BARS} (≈ {FORWARD_BARS * bar_num} {bar_unit})")
    print(f"  Continuous cols : 1  (Close)")
    print(f"  Feature cols    : {len(candidates)}  (auto-detected from FEATURES)")
    print(f"  News sentiment  : {'✓ included' if news_bool_present else '✗ skipped'}")
    print(f"  Total rows      : {total:,}")
    print(f"\n  Class distribution:")
    for code in [BUY, HOLD, SELL]:
        cnt  = (dataset["target"] == code).sum()
        name = LABEL_NAMES[code]
        bar  = "█" * max(1, int(30 * cnt / total))
        print(f"    {name:4s} (code={code})  {cnt:6,}  ({100 * cnt / total:5.1f}%)  {bar}")
    ratio = (dataset["target"] == BUY).sum() / max(1, (dataset["target"] == SELL).sum())
    if ratio > 3 or ratio < 0.33:
        print(f"\n  ⚠  BUY/SELL ratio = {ratio:.2f}. "
              "Consider class weighting or oversampling.")
    print(f"{'─' * 55}")

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION 2  —  UNLABELED DATASET  (unsupervised learning)
# ─────────────────────────────────────────────────────────────────────────────

def build_unlabeled_dataset(
    ticker:   str = TICKER,
    period:   str = PERIOD,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """
    Build a feature-only dataset for unsupervised learning (clustering,
    anomaly detection, dimensionality reduction, etc.).

    Pipeline
    --------
    [1] Download OHLCV
    [2] Build technical features
    [3] Add news sentiment (optional)
    [4] Clean NaN and finalise

    No labelling step — every bar in the download window is kept, including
    the final FORWARD_BARS rows that build_labeled_dataset() must drop.

    Returns
    -------
    pd.DataFrame with columns:
        <all feature columns>  — from feature_engineering.FEATURES + news
        cols, whichever exist
        (no target, no raw OHLCV reference columns)

    The datetime index is preserved so clusters or anomaly scores can be
    traced back to specific bars for inspection alongside price data.

    NaN handling
    ------------
    FVG zones  → 0  (no active zone)
    sentiment  → 0  (no news = neutral)
    warm-up    → ffill then 0-fill
    No rows are dropped — all bars are usable for unsupervised methods.
    """
    print(f"\n{'═' * 55}")
    print(f"  Building UNLABELED dataset  ({ticker})")
    print(f"{'═' * 55}")

    # ── Shared steps [1–3] ───────────────────────────────────────────────────
    enriched, feat_cols = _build_enriched(ticker, period, interval)

    # ── [4] Finalise ──────────────────────────────────────────────────────────
    print("[4/4] Cleaning and finalising …")

    # Keep only the engineered feature columns — no OHLCV, no target
    feat_cols_present = [c for c in feat_cols if c in enriched.columns]
    dataset = enriched[feat_cols_present].copy()

    # Apply shared feature NaN handling (no rows dropped — all bars usable)
    dataset = _clean_features(dataset, feat_cols_present)

    # ── Summary ───────────────────────────────────────────────────────────────
    news_fill = [c for c in _NEWS_COLS if c in dataset.columns]
    print(f"\n{'─' * 55}")
    print(f"  Unlabeled dataset summary")
    print(f"{'─' * 55}")
    print(f"  Ticker          : {ticker}")
    print(f"  Period/Interval : {period} / {interval}")
    print(f"  Feature columns : {len(feat_cols_present)}")
    print(f"  News sentiment  : {'✓ included' if news_fill else '✗ skipped'}")
    print(f"  Total rows      : {len(dataset):,}  (no tail rows dropped)")
    print(f"  NaN remaining   : {dataset.isna().sum().sum()}")
    print(f"{'─' * 55}")

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# FILTER UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def filter_dataset(
    dataset:      pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Return a copy of dataset containing only the requested feature columns.

    Works on both labeled and unlabeled datasets:
      - If 'target' is present it is always kept, regardless of feature_cols.
      - Columns in feature_cols that don't exist in dataset are skipped with
        a warning rather than raising a KeyError.

    Parameters
    ----------
    dataset      : pd.DataFrame
        Output of build_labeled_dataset() or build_unlabeled_dataset().
    feature_cols : list[str]
        The columns you want to keep.  Pass any subset of
        feature_engineering.FEATURES (+ news cols) — same format as the
        FEATURES list in feature_engineering.py.

    Returns
    -------
    pd.DataFrame with the datetime index preserved and columns ordered as:
        [requested feature cols that exist]  +  ['target']  (if present)

    Example
    -------
        from feature_engineering import FEATURES
        filtered = filter_dataset(labeled, FEATURES)
    """
    # Separate out which requested columns actually exist
    present = [c for c in feature_cols if c in dataset.columns]
    missing = [c for c in feature_cols if c not in dataset.columns]

    if missing:
        print(f"[filter_dataset] ⚠  {len(missing)} column(s) not found and skipped:")
        for c in missing:
            print(f"    • {c}")

    # Always append target last if the dataset has it
    keep = list(dict.fromkeys(present))   # deduplicate, preserve order
    if "target" in dataset.columns and "target" not in keep:
        keep.append("target")

    filtered = dataset[keep].copy()

    print(f"[filter_dataset] {len(present)} of {len(feature_cols)} requested "
          f"columns kept  (+target: {'yes' if 'target' in filtered.columns else 'no'})"
          f"  →  {filtered.shape[1]} total columns, {len(filtered):,} rows.")

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)

    # ── Labeled (supervised) ──────────────────────────────────────────────────
    labeled = build_labeled_dataset()
    labeled.to_csv(OUTPUT_PATH)
    print(f"\n✓  Saved labeled   → {OUTPUT_PATH}")

    # ── Unlabeled (unsupervised) ──────────────────────────────────────────────
    unlabeled = build_unlabeled_dataset()
    unlabeled.to_csv(OUTPUT_PATH_UNLABELED)
    print(f"✓  Saved unlabeled → {OUTPUT_PATH_UNLABELED}")

    # ── Filter example — same FEATURES list used everywhere else ─────────────
    # Pass any list of column names to get back a narrowed dataset. Columns
    # that don't exist (e.g. news cols when sentiment is off) are skipped
    # with a warning. target is always preserved automatically.
    # Uncomment/comment entries directly in feature_engineering.FEATURES to
    # control what shows up here — there's no separate list to maintain.
    filtered_labeled   = filter_dataset(labeled,   FEATURES)
    filtered_unlabeled = filter_dataset(unlabeled, FEATURES)

    filtered_labeled.to_csv("filtered_labeled_dataset.csv")
    filtered_unlabeled.to_csv("filtered_unlabeled_dataset.csv")
    print(f"\n✓  Saved filtered labeled   → filtered_labeled_dataset.csv")
    print(f"✓  Saved filtered unlabeled → filtered_unlabeled_dataset.csv")

    # ── Quick peek at both ────────────────────────────────────────────────────
    print(f"\nLabeled — first 3 rows:")
    print(labeled.head(3).to_string())

    print(f"\nUnlabeled — first 3 rows:")
    print(unlabeled.head(3).to_string())

    print(f"\nFiltered labeled — first 3 rows:")
    print(filtered_labeled.head(3).to_string())

if __name__ == "__main__":
    from dataset_builder import build_labeled_dataset, TICKER, PERIOD, INTERVAL
    import yfinance as yf

    # Need raw OHLCV separately since build_labeled_dataset() only returns
    # Close + boolean feature cols + target, not High/Low/Open/Volume.
    raw = yf.download(TICKER, period=PERIOD, interval=INTERVAL,
                       progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    labeled = build_labeled_dataset(TICKER, PERIOD, INTERVAL)

    plot_target_signals(raw, labeled, lookback_bars=300,
                         title=f"{TICKER} — Labeled BUY/SELL Targets")