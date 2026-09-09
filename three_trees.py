import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform

from sklearn.calibration    import CalibratedClassifierCV
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline       import Pipeline
from sklearn.tree           import DecisionTreeClassifier

from dataset_builder import build_labeled_dataset
from force_features import ForcedRootTree


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Data
TICKER          = "SPY"
PERIOD          = "60d"
INTERVAL        = "1m"

# The three timeframes for building fully independent, per-timeframe
# dataframes (see build_multi_timeframe_datasets below).
INTERVALS = ["1m", "5m", "15m"]

# Yahoo Finance enforces hard history caps regardless of what's requested:
#   1m bars      -> roughly the last 7 days only
#   5m/15m bars  -> roughly the last 60 days only
# Requesting more doesn't error — it silently returns less than asked for.
# So each interval gets its own safe period rather than sharing PERIOD.
TIMEFRAME_PERIODS = {
    "1m":  "7d",
    "5m":  "60d",
    "15m": "60d",
}

# Features that mean the same thing regardless of timeframe — each
# per-timeframe dataframe computes its OWN native version of these under
# the same column name (e.g. the 1m dataframe's "is_trending" is ADX-based
# on 1-minute bars; the 5m dataframe's "is_trending" is the same indicator
# computed on 5-minute bars). Add or remove entries here to change what
# EVERY tree sees, in one place.
#
# IMPORTANT: every entry here must be a column that survives
# dataset_builder.py's boolean-only filter (_is_binary_col). Continuous
# indicators — ADX, rsi, macd_line, lr_upper/lower, any raw price level
# like fvg_bull_top, or FVG stack counts (0,1,2,3...) — never make it into
# the dataframe build_labeled_dataset() returns, no matter what's
# uncommented in feature_engineering.FEATURES. Listing one here won't
# error loudly — load_and_prepare() below just silently drops it and warns.
# Keep this in sync with whatever's uncommented (and boolean) in
# feature_engineering.FEATURES.
SHARED_FEATURES = [
    "high_wick_sweep", "low_wick_sweep",
    "isGreen", "isHigh", "isLow",
    "trend_ranging", "trend_uptrend",
    "is_trending", "adx_uptrend", "adx_downtrend",
    "DI_cross_up", "DI_cross_down",
    "trend_strengthening", "trend_weakening",
    "strong_up", "fading_up", "strong_down", "fading_down",
    "near_upper_band", "near_lower_band", "touches_upper", "touches_lower",
    "broke_above", "broke_below", "bands_widening", "bands_narrowing",
    "macd_bullish_entry", "macd_bearish_entry",
    "rsi_overbought", "rsi_oversold",
    "mr_below_sma20", "mr_above_sma20",
    "mr_bb_below_lower", "mr_bb_above_upper",
    "mr_rsi_oversold", "mr_rsi_overbought",
    "mr_z_score_low", "mr_z_score_high",
    "mr_below_vwap", "mr_above_vwap",
    "mo_roc_positive_20", "mo_roc_negative_20",
    "mo_golden_cross", "mo_death_cross",
    "mo_macd_cross_up", "mo_macd_cross_down",
    "mo_adx_trending", "mo_breakout_high20", "mo_breakdown_low20",
    "mo_volume_surge", "mo_consecutive_up3", "mo_consecutive_down3",
    "mo_combo_long", "mo_combo_short",
    "in_bull_fvg", "in_bear_fvg",
    "recent_low_sweep", "recent_high_sweep",
    "setup_bull_sweep_fvg", "setup_bear_sweep_fvg",
    "setup_bull_confirmed", "setup_bear_confirmed",
    "fvg_bull_inversion", "fvg_bear_inversion",
    "fvg_bull_stacking", "fvg_bear_stacking",
    "fvg_targeting_next_bull", "fvg_targeting_next_bear",
    "fvg_at_next_bull", "fvg_at_next_bear",
    "fvg_bull_inv_with_target", "fvg_bear_inv_with_target",
    "idx_div_spy_up_qqq_down", "idx_div_spy_down_qqq_up",
    "idx_div_any", "idx_div_rolling3", "idx_corr_breakdown",

    # ── session / time-of-day (new) ──────────────────────────
    "sess_open_30m", "sess_lunch_lull", "sess_close_30m",

    # ── higher-timeframe trend alignment (new) ───────────────
    "htf_trend_up",

    # ── volatility regime (new) ───────────────────────────────
    "vol_regime_elevated",

    # ── breaker confluence, shared across all 3 trees (new) ──
    "breaker_in_killzone",
    "breaker_near_eq_liquidity",

    "lorentzian_signal_long", 
    "lorentzian_signal_short",
    "lorentzian_bars_since_signal",
    "lorentzian_prediction",
    "nwe_buy_signal",
    "nwe_sell_signal",
    "piv_buy_signal",
    "piv_sell_signal",
]

# Which 2 breaker columns belong to which timeframe's tree, kept separate
# from SHARED_FEATURES since these are the one category that must NOT be
# shared — giving the 1m tree visibility into the 5m/15m breaker columns
# would leak other-timeframe information into what's supposed to be an
# isolated, timeframe-specific signal.
TIMEFRAME_BREAKER_PAIR = {
    "1m":  ["breaker_bull_double_inverse_1m",  "breaker_bear_double_inverse_1m",
            "breaker_bull_favorable_rr_1m", "breaker_bear_favorable_rr_1m",
            "breaker_strong_displacement_1m", "breaker_volume_confirmed_1m",
            "breaker_unicorn_1m", "breaker_first_retest_1m"],
    "5m":  ["breaker_bull_double_inverse_5m",  "breaker_bear_double_inverse_5m",
            "breaker_bull_favorable_rr_5m", "breaker_bear_favorable_rr_5m",
            "breaker_strong_displacement_5m", "breaker_volume_confirmed_5m",
            "breaker_unicorn_5m", "breaker_first_retest_5m"],
    "15m": ["breaker_bull_double_inverse_15m", "breaker_bear_double_inverse_15m",
            "breaker_bull_favorable_rr_15m", "breaker_bear_favorable_rr_15m",
            "breaker_strong_displacement_15m", "breaker_volume_confirmed_15m",
            "breaker_unicorn_15m", "breaker_first_retest_15m"],
}

TIMEFRAME_FEATURES = {
    tf: SHARED_FEATURES + pair
    for tf, pair in TIMEFRAME_BREAKER_PAIR.items()
}

N_SHARED_TREES = 7

def _split_shared_features(features, n_splits):
    """Divide SHARED_FEATURES into n_splits near-equal, non-overlapping chunks."""
    chunks = [[] for _ in range(n_splits)]
    for i, feat in enumerate(features):
        chunks[i % n_splits].append(feat)
    return chunks

SHARED_FEATURE_CHUNKS = _split_shared_features(SHARED_FEATURES, N_SHARED_TREES)

FORCE_COLS = ["_breaker_active", "_lorentzian_active"]

# Split fractions (must sum to 1.0); all chronological — no shuffling
TRAIN_FRAC      = 0.60   # oldest 60 % → hyperparameter tuning
CAL_FRAC        = 0.15   # next  15 % → probability calibration
# TEST_FRAC     = 0.25   # final 25 % → held-out evaluation (implicit)

RANDOM_STATE    = 42
N_ITER          = 40     # unused now that this uses GridSearchCV, not
                          # RandomizedSearchCV — left in case you switch back
CV_SPLITS       = 3      # TimeSeriesSplit folds inside training set

# Each tree now sees SHARED_FEATURES (~80 boolean columns) plus its own
# timeframe's 2 breaker columns — no longer the narrow 2-feature case this
# grid was originally sized for. With this many inputs there's real
# overfitting risk (unlike the old 2-feature case, where every leaf was
# already forced to use both), so max_depth, min_samples_leaf/split, and
# max_features all genuinely matter here.
PARAM_DIST = {
    "criterion"        : ["gini", "entropy"],
    "max_depth"        : [2, 3, 4, 5, 6, None],
    "min_samples_leaf" : [1, 5, 10, 20],
    "min_samples_split": [2, 10, 20],
    "max_features"     : [None],
}

# Feature pruning — not used by this file's per-timeframe trees (each tree
# only has 2 features to begin with, nothing to prune), left here for
# parity with random_forest.py's config in case you fold that back in.
CORR_THRESHOLD  = 0.95
IMP_THRESHOLD   = 0.0

# HOLD undersampling
# All BUY and SELL rows are kept.  Only this fraction of HOLD rows are kept.
# 0.2 = keep 20% of HOLD bars → forces the model to treat BUY/SELL as equally
# common during training without artificially reweighting loss.
TARGET_BS_RATIO = 0.8   

# Any-tree-fires prediction
# If any single tree in the forest predicts BUY or SELL with leaf purity >=
# this threshold, the bar is signalled as BUY/SELL regardless of what the
# majority of trees voted.
TREE_CERTAINTY_THRESHOLD = 0.70

# Standard confidence threshold (used alongside any-tree-fires for comparison)
CONFIDENCE_THRESHOLD = 0.35

# Target codes (must match dataset_builder.py)
BUY  = 2
HOLD = 1
SELL = 0


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def build_multi_timeframe_datasets(ticker=TICKER, intervals=INTERVALS):
    """
    Build three fully independent labeled dataframes, one per timeframe in
    `intervals` — each downloaded and processed natively AT its own
    interval (its own candles, its own breaker-block features, its own
    forward-return labels), not derived or aligned from any other
    timeframe.

    Each timeframe gets its own period from TIMEFRAME_PERIODS, since Yahoo
    Finance enforces hard history caps that differ by interval.

    Note: every dataframe returned here still contains ALL of the breaker
    columns FEATURES currently selects in feature_engineering.py — not
    just that timeframe's own pair. load_and_prepare() below is what
    narrows each one down to its own 2 columns; this function's job is
    just building the three independent, natively-sourced dataframes.

    Returns
    -------
    dict[str, pd.DataFrame] keyed by whatever's in `intervals` (default
    "1m" / "5m" / "15m").
    """
    datasets = {}
    for tf in intervals:
        period = TIMEFRAME_PERIODS.get(tf, PERIOD)
        print(f"\n{'#' * 60}")
        print(f"  Building {tf} dataframe  (ticker={ticker}, period={period})")
        print(f"{'#' * 60}")
        datasets[tf] = build_labeled_dataset(ticker=ticker, period=period, interval=tf)

    return datasets

def train_shared_tree(df, feature_cols, label):
    """Train a single DecisionTree on the given feature subset of the
    PRIMARY_TF dataset (shared trees don't need their own timeframe —
    they all read off the same dataframe as PRIMARY_TF)."""
    cols = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[{label}] ⚠ missing from df: {missing}")
    if not cols:
        raise ValueError(
            f"[{label}] no requested columns found in df. "
            f"Requested: {feature_cols}. df has {len(df.columns)} cols total."
        )

    X = df[cols].copy()
    y = df["target"]

    X_train, X_cal, X_test, y_train, y_cal, y_test = chronological_split(X, y)
    X_train, y_train = undersample_hold(X_train, y_train, target_bs_ratio=TARGET_BS_RATIO)

    tree = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=tree, param_grid=PARAM_DIST,
        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        scoring="f1_macro", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    print(f"\n{'=' * 60}")
    print(f"  Shared tree: {label}  ({len(cols)} features)")
    print(f"{'=' * 60}")
    print(f"  Features           : {cols}")
    print(f"  Best params        : {grid.best_params_}")
    print(f"  Best CV F1 (macro) : {grid.best_score_:.4f}")
    print(f"  Test accuracy      : {grid.score(X_test, y_test):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    full_pred = pd.Series(grid.predict(X), index=X.index)
    return grid, y_pred, y_test, X_test.index, full_pred

def combine_signals(*preds_list):
    """Majority vote across N trees' predictions (already aligned)."""
    preds = np.stack(preds_list, axis=1)
    buy_count  = (preds == BUY).sum(axis=1)
    sell_count = (preds == SELL).sum(axis=1)

    signal = np.full(preds.shape[0], HOLD, dtype=int)
    signal[buy_count  > sell_count] = BUY
    signal[sell_count > buy_count]  = SELL
    return signal


def load_and_prepare(df, interval=INTERVAL):
    print(f"── Loading dataset ({interval}) ──────────────────────────")

    requested = TIMEFRAME_FEATURES.get(interval, [])
    features  = [c for c in requested if c in df.columns]
    missing   = [c for c in requested if c not in df.columns]

    if missing:
        print(f"  ⚠  {len(missing)} requested column(s) not found and skipped: {missing}")
    if not features:
        raise ValueError(
            f"No columns found for interval='{interval}'. Expected "
            f"{requested} — check FEATURES in feature_engineering.py."
        )

    X = df[features].copy()
    y = df["target"]

    print(f"   Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} cols")
    print(f"   Features used  : {list(X.columns)}")
    print(f"   Class counts   : {dict(y.value_counts().sort_index())}")
    return X, y


def chronological_split(X, y):
    """
    Split X and y into three non-overlapping chronological slices:
        train  → first TRAIN_FRAC of rows  (hyperparameter tuning)
        cal    → next  CAL_FRAC of rows    (probability calibration)
        test   → final remaining rows      (held-out evaluation)

    Why not train_test_split?
    -------------------------
    train_test_split shuffles by default.  On time-series data that means
    future bars appear in the training set, which is lookahead bias — the
    model learns from information it could never have had at prediction time.
    Using a hard chronological cutoff prevents this entirely.
    """
    n      = len(X)
    i_cal  = int(n * TRAIN_FRAC)
    i_test = int(n * (TRAIN_FRAC + CAL_FRAC))

    X_train, y_train = X.iloc[:i_cal],       y.iloc[:i_cal]
    X_cal,   y_cal   = X.iloc[i_cal:i_test], y.iloc[i_cal:i_test]
    X_test,  y_test  = X.iloc[i_test:],      y.iloc[i_test:]

    print(f"\n── Chronological split ──────────────────────────────────")
    for name, ys in [("Train", y_train), ("Cal", y_cal), ("Test", y_test)]:
        counts = dict(ys.value_counts().sort_index())
        print(f"   {name:5s} : {len(ys):5,} rows   classes {counts}")

    return X_train, X_cal, X_test, y_train, y_cal, y_test


def undersample_hold(X_train, y_train, target_bs_ratio=0.8):
    """
    Keep every BUY/SELL row. Sample HOLD rows so that BUY+SELL make up
    target_bs_ratio of the final training set (default 80%), i.e.
    HOLD makes up (1 - target_bs_ratio) (default 20%).

        n_hold_keep = n_buy_sell * (1 - target_bs_ratio) / target_bs_ratio

    If there aren't enough HOLD rows to hit that ratio, keep all of them
    (can't oversample here — ratio then skews more toward BUY/SELL than requested).
    """
    rng = np.random.default_rng(RANDOM_STATE)

    buy_sell_idx = y_train[y_train != HOLD].index
    hold_idx     = y_train[y_train == HOLD].index

    n_bs = len(buy_sell_idx)
    n_hold_target = int(round(n_bs * (1 - target_bs_ratio) / target_bs_ratio))
    n_keep = min(n_hold_target, len(hold_idx))

    kept_hold = rng.choice(hold_idx, size=n_keep, replace=False) if n_keep > 0 else np.array([], dtype=hold_idx.dtype)

    keep_idx = buy_sell_idx.tolist() + kept_hold.tolist()

    X_out = X_train.loc[keep_idx].sort_index()
    y_out = y_train.loc[keep_idx].sort_index()

    print(f"\n── HOLD undersampling (target BUY+SELL ratio={target_bs_ratio}) ──")
    print(f"   Before : {len(y_train):,} rows  {dict(y_train.value_counts().sort_index())}")
    print(f"   After  : {len(y_out):,} rows  {dict(y_out.value_counts().sort_index())}")
    actual_ratio = n_bs / len(y_out) if len(y_out) else 0
    print(f"   Actual BUY+SELL ratio: {actual_ratio:.3f}")
    return X_out, y_out


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# COMBINING THE THREE TREES INTO ONE SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

# Which timeframe's test period sets the cadence the combined signal is
# reported on. The 1m/5m/15m trees each live on a different native
# timestamp axis (different row counts, different date ranges — 1m is
# capped to ~7 days by Yahoo, 5m/15m get ~60), so predictions can't just be
# zipped together positionally; they need a shared reference axis to align
# onto. 5m is a reasonable middle-ground default; change this if you'd
# rather report on 1m's or 15m's cadence instead.
PRIMARY_TF = "1m"


def align_predictions_to_primary(pred_series: pd.Series, primary_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Point-in-time (no lookahead) alignment of one tree's full-history
    prediction series onto a different timestamp axis, via a backward
    as-of merge: for each timestamp in `primary_index`, take that tree's
    most recent prediction at or before that moment. Timestamps in
    `primary_index` earlier than any of pred_series's own timestamps
    fall back to HOLD (nothing to align to yet).
    """
    s = pred_series.sort_index().copy()
    s.index.name = "ts"
    left = pd.DataFrame({"ts": pd.DatetimeIndex(primary_index)}).sort_values("ts")
    merged = pd.merge_asof(left, s.reset_index(name="pred"), on="ts", direction="backward")
    merged = merged.set_index("ts").reindex(pd.DatetimeIndex(primary_index))
    return merged["pred"].fillna(HOLD).astype(int).values


# def combine_signals(pred_1m: np.ndarray, pred_5m: np.ndarray, pred_15m: np.ndarray) -> np.ndarray:
#     """
#     Combine three trees' per-bar predictions (already aligned onto the same
#     timestamp axis) into one signal.

#     Rule
#     ----
#     buy_count  = how many of the three trees predict BUY
#     sell_count = how many of the three trees predict SELL

#         signal = BUY   if buy_count  > sell_count
#         signal = SELL  if sell_count > buy_count
#         signal = HOLD  otherwise (a tie, including 0-0 or 1-1)

#     This is exactly the rule as specified — "one buy and zero sells, or two
#     buys -> BUY" (mirrored for SELL) — just written as a direct comparison:
#     buy_count(1) > sell_count(0) covers the first case, buy_count(2) beats
#     any sell_count <= 1 covers the second. The same comparison naturally
#     extends to cases not spelled out explicitly (three buys; two buys with
#     one sell) the same way — larger count wins, matching count is a tie.
#     """
#     preds = np.stack([pred_1m, pred_5m, pred_15m], axis=1)
#     buy_count  = (preds == BUY).sum(axis=1)
#     sell_count = (preds == SELL).sum(axis=1)

#     signal = np.full(len(pred_1m), HOLD, dtype=int)
#     signal[buy_count  > sell_count] = BUY
#     signal[sell_count > buy_count]  = SELL
#     return signal

def plot_tree_feature_importances(grids, output_path="tree_feature_importances.png"):
    """
    One horizontal bar-chart subplot per timeframe tree, showing each
    tree's feature_importances_ sorted descending. Reads straight off
    grids[tf].best_estimator_ — no retraining needed.
    """
    tfs = list(grids.keys())
    fig, axes = plt.subplots(1, len(tfs), figsize=(6 * len(tfs), 8))
    if len(tfs) == 1:
        axes = [axes]

    for ax, tf in zip(axes, tfs):
        tree = grids[tf].best_estimator_
        importances = pd.Series(tree.feature_importances_, index=tree.feature_names_in_)
        importances = importances[importances > 0].sort_values(ascending=True)

        ax.barh(importances.index, importances.values, color="#009988")
        ax.set_title(f"{tf} tree — feature importances")
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓  Saved feature importance chart → {output_path}")
    return output_path

def balance_classes_equal(X_train, y_train):
    """
    Downsample all three classes (BUY, SELL, HOLD) to the same size — the
    count of whichever class has the fewest rows — so the model sees
    ~33% of each class in training.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    idx_by_class = {
        cls: y_train[y_train == cls].index
        for cls in [BUY, SELL, HOLD]
    }
    counts_before = {cls: len(idx) for cls, idx in idx_by_class.items()}
    n_min = min(counts_before.values())

    if n_min == 0:
        raise ValueError(
            f"At least one class has zero training rows after your "
            f"labelling threshold: {counts_before}. Lower the threshold "
            f"or widen FORWARD_BARS — there's nothing to balance to."
        )

    keep_idx = []
    for cls, idx in idx_by_class.items():
        if len(idx) > n_min:
            # FIX: sample positions (ints), not the DatetimeIndex itself —
            # rng.choice on a DatetimeIndex silently corrupts the values
            # (converts to plain datetime.datetime, which then fails to
            # match X_train's pd.Timestamp index in .loc[]).
            positions = rng.choice(len(idx), size=n_min, replace=False)
            sampled = idx[positions]
        else:
            sampled = idx
        keep_idx.extend(sampled.tolist())

    X_out = X_train.loc[keep_idx].sort_index()
    y_out = y_train.loc[keep_idx].sort_index()

    print(f"\n── Equal-class balancing (target ~33% each) ──")
    print(f"   Before : {dict(y_train.value_counts().sort_index())}")
    print(f"   After  : {dict(y_out.value_counts().sort_index())}   (n_min={n_min} per class)")
    return X_out, y_out


def build_model(ticker=TICKER, period=PERIOD, intervals=INTERVALS):
    """
    End-to-end pipeline, run independently for each timeframe:
        1. Build the timeframe's own native dataframe
        2. Narrow to that timeframe's own 2 breaker columns
        3. Chronological train / cal / test split
        4. HOLD undersampling on the training split
        5. Grid-search a DecisionTreeClassifier (class_weight='balanced',
           scored on f1_macro so the imbalanced classes actually count,
           TimeSeriesSplit CV so no future bars leak into tuning)
        6. Evaluate on the held-out test split — accuracy, classification
           report, and confusion matrix

    Returns
    -------
    grids       : dict[str, GridSearchCV]   — fitted search object per timeframe
    preds       : dict[str, np.ndarray]     — test-set predictions per timeframe
    tests       : dict[str, pd.Series]      — test-set true labels per timeframe
    combined    : np.ndarray                — the 3-tree combined signal,
                  reported on PRIMARY_TF's test timestamps
    """
    datasets = build_multi_timeframe_datasets(ticker=ticker, intervals=intervals)

    grids, preds, tests = {}, {}, {}
    full_preds = {}     # tf -> pd.Series of predictions across that tf's ENTIRE
                          # dataframe (not just its test slice) — needed so the
                          # other two timeframes have something to align onto
    test_index = {}     # tf -> the DatetimeIndex of that tf's own test slice

    FORCE_COLS_CANDIDATES = ["_breaker_active", "_lorentzian_active"]

    for tf in intervals:
        X, y = load_and_prepare(datasets[tf], interval=tf)
        X_train, X_cal, X_test, y_train, y_cal, y_test = chronological_split(X, y)
        X_train, y_train = undersample_hold(X_train, y_train, target_bs_ratio=TARGET_BS_RATIO)  # was: undersample_hold(...)

        tree = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
        grid = GridSearchCV(
            estimator=tree,
            param_grid=PARAM_DIST,
            cv=TimeSeriesSplit(n_splits=CV_SPLITS),
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

        print(f"\n{'=' * 60}")
        print(f"  {tf} tree")
        print(f"{'=' * 60}")
        print(f"  Best params        : {grid.best_params_}")
        print(f"  Best CV F1 (macro) : {grid.best_score_:.4f}")
        print(f"  Test accuracy      : {grid.score(X_test, y_test):.4f}")
        print(f"\n  Classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        labels_present = sorted(set(y_test) | set(y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=labels_present)
        print(f"  Confusion matrix (rows=true {labels_present}, cols=pred {labels_present}):")
        print(cm)

        grids[tf]      = grid
        preds[tf]      = y_pred
        tests[tf]      = y_test
        test_index[tf] = X_test.index

        # Full-history predictions (train+cal+test combined), so the other
        # two timeframes have a complete series to align onto — not just
        # this tree's own test window.
        full_preds[tf] = pd.Series(grid.predict(X), index=X.index)

        # ── Train 7 additional shared-feature-only trees on PRIMARY_TF's data ────
    primary_df = datasets[PRIMARY_TF]
    print(f"\n[debug] primary_df columns ({len(primary_df.columns)}): {sorted(primary_df.columns)}")
    print(f"[debug] SHARED_FEATURES ({len(SHARED_FEATURES)}): {SHARED_FEATURES}")
    print(f"[debug] shared_1 chunk: {SHARED_FEATURE_CHUNKS[0]}")
    shared_full_preds = []
    for i, chunk in enumerate(SHARED_FEATURE_CHUNKS):
        _, _, _, _, full_pred = train_shared_tree(
            primary_df, chunk, label=f"shared_{i+1}"
        )
        shared_full_preds.append(full_pred)

    # ── Combine the three per-timeframe trees onto PRIMARY_TF's test cadence ─
    primary_index = test_index[PRIMARY_TF]
    aligned = {
        tf: (full_preds[tf].reindex(primary_index).ffill().fillna(HOLD).astype(int).values
             if tf == PRIMARY_TF
             else align_predictions_to_primary(full_preds[tf], primary_index))
        for tf in intervals
    }
    shared_aligned = [
        s.reindex(primary_index).ffill().fillna(HOLD).astype(int).values
        for s in shared_full_preds
    ]

    combined = combine_signals(aligned["1m"], aligned["5m"], aligned["15m"], *shared_aligned)

    # ── Combine the three trees onto PRIMARY_TF's test cadence ───────────────
    primary_index = test_index[PRIMARY_TF]
    aligned = {
        tf: (full_preds[tf].reindex(primary_index).ffill().fillna(HOLD).astype(int).values
             if tf == PRIMARY_TF
             else align_predictions_to_primary(full_preds[tf], primary_index))
        for tf in intervals
    }
    combined = combine_signals(aligned["1m"], aligned["5m"], aligned["15m"])

    y_true_primary = tests[PRIMARY_TF].values
    print(f"\n{'=' * 60}")
    print(f"  Combined signal (3-tree vote, reported on {PRIMARY_TF} cadence)")
    print(f"{'=' * 60}")
    print(f"  Accuracy : {(combined == y_true_primary).mean():.4f}")
    print(f"\n  Classification report:")
    print(classification_report(y_true_primary, combined, zero_division=0))

    labels_present = sorted(set(y_true_primary) | set(combined))
    cm = confusion_matrix(y_true_primary, combined, labels=labels_present)
    print(f"  Confusion matrix (rows=true {labels_present}, cols=pred {labels_present}):")
    print(cm)

    return grids, preds, tests, combined


if __name__ == "__main__":
    build_model()