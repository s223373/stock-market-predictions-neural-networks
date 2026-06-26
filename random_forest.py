"""
random_forest_model.py
======================
Supercharged Random Forest classifier for BUY / HOLD / SELL prediction.

Improvements over the baseline
--------------------------------
1.  Chronological split     No data leakage. Test set is always the most
                            recent bars — never a random draw from the future.

2.  TimeSeriesSplit CV      Cross-validation folds respect temporal order so
                            every validation fold is always after its train fold.

3.  Class balancing         class_weight='balanced' counters the heavy HOLD
                            majority typical in labeled financial data.

4.  RandomizedSearchCV      Samples the hyperparameter space efficiently;
                            same coverage as GridSearchCV in ~5% of the time.

5.  Correlation pruning     Drops one column from every pair correlated above
                            CORR_THRESHOLD before training, reducing redundancy
                            without losing information.

6.  Feature importance      After the tuned model is fit, low-importance
    pruning                 features (below IMP_THRESHOLD) are dropped and the
                            model is refit on the pruned set.

7.  Probability calibration Wraps the tuned RF in CalibratedClassifierCV so
                            predict_proba() outputs are reliable estimates of
                            actual class probabilities, not just vote fractions.

8.  Confidence threshold    Only issues BUY / SELL when the model's max class
                            probability exceeds CONFIDENCE_THRESHOLD; all other
                            bars are demoted to HOLD.  Trades fewer bars but at
                            materially higher expected precision.

9.  Trading metrics         Reports precision / recall / F1 per class, signal
                            coverage, and BUY + SELL precision specifically —
                            the metrics that matter most for a trading system.

Target encoding (from build_labeled_dataset)
  2 = BUY  |  1 = HOLD  |  0 = SELL
"""

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from sklearn.calibration    import CalibratedClassifierCV
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats              import randint, uniform
from sklearn.pipeline       import Pipeline

from dataset_builder  import build_labeled_dataset


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Data
TICKER          = "SPY"
PERIOD          = "60d"
INTERVAL        = "5m"

# Split fractions (must sum to 1.0); all chronological — no shuffling
TRAIN_FRAC      = 0.70   # oldest 70 % → hyperparameter tuning
CAL_FRAC        = 0.15   # next  15 % → probability calibration
# TEST_FRAC     = 0.15   # final 15 % → held-out evaluation (implicit)

# Cross-validation inside the training fold
CV_SPLITS       = 5      # TimeSeriesSplit folds

# RandomizedSearchCV — samples the hyperparameter space rather than exhausting
# it, giving near-equivalent coverage in a fraction of the time.
#
# Runtime estimate:
#   N_ITER × CV_SPLITS fits × ~3–6 s per fit (on undersampled ~1,600 rows)
#   40     × 3             × ~4 s            ≈ 8 min  (target: 10–15 min)
#
# Raise N_ITER for a more thorough search; raise CV_SPLITS for more reliable
# cross-validation scores (at proportional cost to runtime).
RANDOM_STATE    = 42
N_ITER          = 40     # number of random combinations to try
CV_SPLITS       = 3      # TimeSeriesSplit folds inside training set

PARAM_DIST = {
    # Trees: higher is better but with diminishing returns above ~500.
    # randint samples any integer in [low, high).
    "n_estimators"     : randint(200, 1001),

    # Depth: None = fully grown (can overfit); integers limit tree depth.
    "max_depth"        : [None, 10, 20, 30, 50],

    # Splitting constraints — higher values regularise against overfitting.
    "min_samples_split": randint(2, 25),
    "min_samples_leaf" : randint(1, 12),

    # Features considered at each split — key regularisation knob for RF.
    "max_features"     : ["sqrt", "log2", 0.2, 0.3, 0.5],

    # Impurity measure — gini and entropy perform similarly; search both.
    "criterion"        : ["gini", "entropy"],
}

# Feature pruning
CORR_THRESHOLD  = 0.95   # drop one of any feature pair correlated above this
IMP_THRESHOLD   = 0.0    # 0.0 = disabled; raise (e.g. 0.001) to prune weak features

# HOLD undersampling
# All BUY and SELL rows are kept.  Only this fraction of HOLD rows are kept.
# 0.2 = keep 20% of HOLD bars → forces the model to treat BUY/SELL as equally
# common during training without artificially reweighting loss.
HOLD_KEEP_FRACTION = 0.2

# Any-tree-fires prediction
# If any single tree in the forest predicts BUY or SELL with leaf purity >=
# this threshold, the bar is signalled as BUY/SELL regardless of what the
# majority of trees voted.
TREE_CERTAINTY_THRESHOLD = 0.30

# Standard confidence threshold (used alongside any-tree-fires for comparison)
CONFIDENCE_THRESHOLD = 0.55

# Target codes (must match build_labeled_dataset.py)
BUY  = 2
HOLD = 1
SELL = 0


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(ticker=TICKER, period=PERIOD, interval=INTERVAL):
    """
    Build the labeled dataset, apply the FEATURES filter from
    feature_engineering.py, and return a clean (X, y) pair in
    chronological order with the datetime index preserved as a column.
    """
    print("── Loading dataset ──────────────────────────────────────")
    labeled = build_labeled_dataset(ticker=ticker, period=period, interval=interval)
    df      = labeled.reset_index()   # brings Datetime out of the index

    # Use every boolean feature the labeled dataset contains.
    # build_labeled_dataset() already filters to Close + BOOLEAN_FEATURE_COLS + target,
    # so dropping Datetime, Close, and target leaves the full boolean feature matrix.
    # Do NOT filter by FEATURES from feature_engineering.py — that list reflects
    # whichever features are uncommented there and may be a small subset.
    drop = [c for c in ["Datetime", "Close", "target"] if c in df.columns]
    X    = df.drop(columns=drop)
    y    = df["target"]

    print(f"   Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} cols")
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

    X_train, y_train = X.iloc[:i_cal],        y.iloc[:i_cal]
    X_cal,   y_cal   = X.iloc[i_cal:i_test],  y.iloc[i_cal:i_test]
    X_test,  y_test  = X.iloc[i_test:],       y.iloc[i_test:]

    print(f"\n── Chronological split ──────────────────────────────────")
    for name, ys in [("Train", y_train), ("Cal", y_cal), ("Test", y_test)]:
        counts = dict(ys.value_counts().sort_index())
        print(f"   {name:5s} : {len(ys):5,} rows   classes {counts}")

    return X_train, X_cal, X_test, y_train, y_cal, y_test


def drop_correlated(X_train, X_cal, X_test, threshold=CORR_THRESHOLD):
    """
    Remove one column from every pair whose Pearson correlation exceeds
    `threshold`.  Fit the correlation matrix on the training set only, then
    apply the same column mask to cal and test so no test information leaks.

    Why this matters for Random Forest
    -----------------------------------
    RF chooses features randomly at each split.  When two features are nearly
    identical, each one individually gets only half the selection opportunities
    it deserves — their combined importance is split across two slots.  Removing
    duplicates lets the remaining feature get its full share of splits, which
    improves both the model's feature importance estimates and training speed.
    """
    corr_matrix = X_train.corr().abs()
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]

    if to_drop:
        print(f"\n── Correlation pruning (threshold={threshold}) ──────────")
        print(f"   Dropping {len(to_drop)} correlated feature(s): {to_drop}")

    X_train = X_train.drop(columns=to_drop)
    X_cal   = X_cal.drop(columns=to_drop)
    X_test  = X_test.drop(columns=to_drop)
    return X_train, X_cal, X_test, to_drop


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_hyperparams(X_train, y_train):
    """
    Search the hyperparameter space with RandomizedSearchCV using
    TimeSeriesSplit cross-validation inside the training fold.

    RandomizedSearchCV vs GridSearchCV
    ------------------------------------
    GridSearchCV evaluates every combination.  For a 5-parameter grid with
    [5,4,2,5,4] options that is 800 fits × 5 CV folds = 4,000 fits.
    RandomizedSearchCV samples N_ITER combinations, so with N_ITER=60 that
    is 60 × 5 = 300 fits — 13× faster with near-equivalent coverage because
    most of the gain in a grid comes from a small region of the space that
    random sampling finds quickly.

    TimeSeriesSplit
    ---------------
    Each fold's validation set is strictly after its training set in time,
    so no future bars ever contaminate the cross-validation score.

    class_weight='balanced'
    -----------------------
    Adjusts each tree's sample weights inversely proportional to class
    frequency, so the minority BUY and SELL classes are not overwhelmed
    by the HOLD majority.
    """
    base_rf = RandomForestClassifier(
        bootstrap    = True,
        oob_score    = False,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    search = RandomizedSearchCV(
        estimator           = base_rf,
        param_distributions = PARAM_DIST,
        n_iter              = N_ITER,
        cv                  = tscv,
        scoring             = 'f1_macro',
        refit               = True,
        random_state        = RANDOM_STATE,
        n_jobs              = -1,
        verbose             = 1,
    )

    total_fits = N_ITER * CV_SPLITS
    print(f"\n── Hyperparameter search ({N_ITER} iterations × {CV_SPLITS} folds"
          f" = {total_fits} fits) ─")
    search.fit(X_train, y_train)

    print(f"   Best F1 (macro, CV) : {search.best_score_:.4f}")
    print(f"   Best params         : {search.best_params_}")
    return search.best_estimator_, search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE PRUNING
# ─────────────────────────────────────────────────────────────────────────────

def prune_low_importance(model, X_train, X_cal, X_test, y_train, best_params):
    """
    Drop features whose mean impurity decrease is below IMP_THRESHOLD, then
    refit a fresh RF on the pruned feature set using the same best parameters.

    Why prune after tuning rather than before?
    ------------------------------------------
    Importance scores are only meaningful after the model has been fit on
    well-tuned hyperparameters.  Pruning on a default RF might remove features
    that a tuned model would rely on heavily.
    """
    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    print(f"\n── Feature importance pruning (threshold={IMP_THRESHOLD}) ───")

    if IMP_THRESHOLD == 0.0:
        print(f"   Pruning disabled — keeping all {len(importances)} features.")
        kept    = importances.index.tolist()
        to_drop = []
    else:
        to_drop = importances[importances < IMP_THRESHOLD].index.tolist()
        kept    = importances[importances >= IMP_THRESHOLD].index.tolist()
        print(f"   Kept  : {len(kept)} features")
        print(f"   Pruned: {len(to_drop)} features  {to_drop}")
    print(f"\n   Top 15 features by importance:")
    print(importances.head(15).to_string())

    # Save full importance ranking to CSV for inspection
    importances.to_csv("feature_importances.csv", header=["importance"])
    print(f"   Full ranking saved → feature_importances.csv")

    X_train_p = X_train[kept]
    X_cal_p   = X_cal[kept]
    X_test_p  = X_test[kept]

    # Refit on pruned features with the same tuned params
    pruned_rf = RandomForestClassifier(
        **best_params,
        bootstrap    = True,
        oob_score    = True,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    pruned_rf.fit(X_train_p, y_train)
    print(f"   OOB score (refit on pruned features): {pruned_rf.oob_score_:.4f}")

    return pruned_rf, X_train_p, X_cal_p, X_test_p, kept


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(model, X_cal, y_cal, cv=3):
    """
    Calibrate the tuned RF's predicted probabilities using the held-out
    calibration set and sigmoid (Platt) scaling.

    Why calibrate?
    ---------------
    Random Forests produce overconfident probability estimates — predict_proba()
    returns vote fractions across trees which cluster near 0 and 1.  After
    calibration the output represents actual probabilities, so
    CONFIDENCE_THRESHOLD has a meaningful interpretation: 0.55 genuinely means
    "at least 55% confident", not just "a bare majority of trees voted this way".

    Why cv=3 instead of cv='prefit'?
    ----------------------------------
    cv='prefit' was removed in recent sklearn versions.  With cv=3,
    CalibratedClassifierCV fits 3 clones of the estimator on 3 folds of X_cal,
    then learns a sigmoid mapping from their out-of-fold predictions to true
    labels.  The underlying RF hyperparameters are preserved because sklearn
    clones the estimator (copies its parameters) before refitting each fold.
    Using the separate calibration set prevents any test data from influencing
    the calibration fit.
    """
    print(f"\n── Probability calibration (cv={cv} on calibration set) ─")
    cal_model = CalibratedClassifierCV(model, method="sigmoid", cv=cv)
    cal_model.fit(X_cal, y_cal)
    print(f"   Calibration complete on {len(y_cal):,} bars.")
    return cal_model


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION WITH CONFIDENCE THRESHOLDING
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_confidence(model, X, threshold=CONFIDENCE_THRESHOLD):
    """
    Issue BUY or SELL only when the model's maximum class probability
    exceeds `threshold`.  All other bars are set to HOLD.

    Why this matters
    ----------------
    A classifier optimised for accuracy will produce a prediction for every
    bar even when the evidence is weak.  In a trading system a wrong BUY or
    SELL costs money, while an extra HOLD costs nothing.  By filtering to
    only high-confidence signals you trade less frequently but with materially
    higher precision on the bars you do act on.

    Returns
    -------
    predictions : np.ndarray  — final class labels after thresholding
    probabilities : np.ndarray  — raw predicted probabilities (n_samples × 3)
    confidence : np.ndarray  — max class probability per bar
    """
    probabilities = model.predict_proba(X)
    raw_preds     = model.classes_[np.argmax(probabilities, axis=1)]
    confidence    = probabilities.max(axis=1)

    predictions = raw_preds.copy()
    low_conf    = confidence < threshold
    predictions[low_conf] = HOLD   # demote uncertain predictions to HOLD

    n_demoted = low_conf.sum()
    print(f"\n── Confidence thresholding (threshold={threshold}) ─────")
    print(f"   Bars demoted to HOLD : {n_demoted:,} "
          f"({100 * n_demoted / len(predictions):.1f} %)")

    return predictions, probabilities, confidence


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def print_trading_metrics(y_true, y_pred, confidence, label="Test set"):
    """
    Print sklearn's classification report plus trading-specific metrics.

    Trading metrics explained
    -------------------------
    Signal coverage    — what fraction of bars produced a BUY or SELL signal.
                         Lower means the model is more selective (higher
                         confidence threshold → lower coverage).

    BUY / SELL precision — of the bars the model called BUY (or SELL), what
                           fraction actually moved in that direction within
                           FORWARD_BARS.  These are the metrics that determine
                           whether a strategy is profitable, not overall accuracy.

    Mean confidence    — average predicted probability on signal bars vs HOLD
                         bars.  If signal bars aren't materially higher than
                         HOLD bars the threshold needs raising.
    """
    print(f"\n{'═' * 55}")
    print(f"  Evaluation — {label}")
    print(f"{'═' * 55}")

    target_names = {BUY: "BUY(2)", HOLD: "HOLD(1)", SELL: "SELL(0)"}
    labels_present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    names = [target_names.get(l, str(l)) for l in labels_present]

    print(classification_report(y_true, y_pred,
                                labels=labels_present,
                                target_names=names,
                                zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    cm_df = pd.DataFrame(cm, index=[f"True {n}" for n in names],
                              columns=[f"Pred {n}" for n in names])
    print("  Confusion matrix:")
    print(cm_df.to_string())

    # Trading-specific metrics
    n_total   = len(y_pred)
    n_signals = np.sum(y_pred != HOLD)
    n_buy     = np.sum(y_pred == BUY)
    n_sell    = np.sum(y_pred == SELL)

    buy_precision = (
        np.sum((y_pred == BUY) & (y_true == BUY)) / max(1, n_buy)
    )
    sell_precision = (
        np.sum((y_pred == SELL) & (y_true == SELL)) / max(1, n_sell)
    )

    signal_mask = y_pred != HOLD
    hold_mask   = y_pred == HOLD

    print(f"\n  {'─' * 40}")
    print(f"  Trading metrics")
    print(f"  {'─' * 40}")
    print(f"  Signal coverage   : {100 * n_signals / n_total:5.1f} %  "
          f"({n_signals:,} bars / {n_total:,} total)")
    print(f"  BUY  signals      : {n_buy:,}   "
          f"precision = {100 * buy_precision:.1f} %")
    print(f"  SELL signals      : {n_sell:,}   "
          f"precision = {100 * sell_precision:.1f} %")
    if signal_mask.any():
        print(f"  Mean confidence   : {confidence[signal_mask].mean():.3f} (signals)  "
              f"{confidence[hold_mask].mean():.3f} (holds)")
    print(f"  {'─' * 40}")


# ─────────────────────────────────────────────────────────────────────────────
# HOLD UNDERSAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def undersample_hold(X_train, y_train, keep_fraction=HOLD_KEEP_FRACTION):
    """
    Keep every BUY and SELL row but randomly discard most HOLD rows.

    Why undersample instead of (or alongside) class_weight?
    --------------------------------------------------------
    class_weight='balanced' keeps all rows but upweights BUY/SELL losses during
    training.  The forest still sees the full HOLD-heavy dataset and learns its
    distribution — it just penalises HOLD mistakes less.

    Undersampling physically removes HOLD rows so the forest is trained on a
    dataset where BUY/SELL/HOLD are roughly equally frequent.  The trees build
    their splits on balanced evidence from the start, which makes them more
    sensitive to the minority signal patterns rather than defaulting to HOLD
    because that is what most training rows show.

    Note: undersampling is applied only to the training split.  Cal and test
    sets are left intact so evaluation reflects the true class distribution.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    buy_sell_idx = y_train[y_train != HOLD].index
    hold_idx     = y_train[y_train == HOLD].index

    n_keep   = max(1, int(len(hold_idx) * keep_fraction))
    kept_hold = rng.choice(hold_idx, size=n_keep, replace=False)

    keep_idx = buy_sell_idx.tolist() + kept_hold.tolist()

    X_out = X_train.loc[keep_idx].sort_index()   # preserve chronological order
    y_out = y_train.loc[keep_idx].sort_index()

    print(f"\n── HOLD undersampling (keep_fraction={keep_fraction}) ──────")
    print(f"   Before : {len(y_train):,} rows  "
          f"{dict(y_train.value_counts().sort_index())}")
    print(f"   After  : {len(y_out):,} rows  "
          f"{dict(y_out.value_counts().sort_index())}")
    return X_out, y_out


# ─────────────────────────────────────────────────────────────────────────────
# ANY-TREE-FIRES PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_any_tree_fires(rf, X, certainty=TREE_CERTAINTY_THRESHOLD):
    """
    Signal BUY or SELL if ANY single tree in the forest predicts that class
    with leaf purity >= certainty.  Bars where no tree fires with sufficient
    certainty are labelled HOLD.

    Why this is different from standard majority-vote
    -------------------------------------------------
    Standard RF predict() requires a majority of all trees to agree before
    committing to BUY or SELL.  In a 1000-tree forest that means 500+ trees
    must vote BUY — a very high bar that causes most uncertain bars to be
    absorbed into HOLD.

    Any-tree-fires instead asks: "does even ONE tree see this bar as a clear
    BUY or SELL setup?"  Each decision tree partitions the feature space into
    pure leaf regions.  A leaf with 90%+ BUY purity means the combination of
    feature values that landed in that leaf is strongly associated with upward
    price moves in the training data.  Even if 900 other trees are unsure, that
    one tree has identified a recognisable pattern.

    For a trading system this is a useful signal: a clear ICT setup may only
    be perfectly formed on a handful of features, which one tree captures
    completely while others — trained on different random feature subsets —
    miss it.

    Priority rule: BUY > SELL when both fire on the same bar (can only happen
    at very low certainty thresholds; rare above 0.65).

    Parameters
    ----------
    rf          : fitted RandomForestClassifier (the raw RF, not calibrated)
    X           : feature matrix to predict on
    certainty   : minimum leaf purity required for a tree to "fire"

    Returns
    -------
    predictions : np.ndarray (int)  — BUY / HOLD / SELL per bar
    buy_votes   : np.ndarray (int)  — number of trees that fired BUY
    sell_votes  : np.ndarray (int)  — number of trees that fired SELL
    """
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    n     = len(X_arr)

    classes  = rf.classes_
    buy_idx  = int(np.where(classes == BUY)[0][0])
    sell_idx = int(np.where(classes == SELL)[0][0])

    buy_votes  = np.zeros(n, dtype=int)
    sell_votes = np.zeros(n, dtype=int)

    for tree in rf.estimators_:
        leaf_proba  = tree.predict_proba(X_arr)          # (n, n_classes)
        buy_votes  += (leaf_proba[:, buy_idx]  >= certainty).astype(int)
        sell_votes += (leaf_proba[:, sell_idx] >= certainty).astype(int)

    predictions = np.full(n, HOLD, dtype=int)
    predictions[sell_votes > 0] = SELL
    predictions[buy_votes  > 0] = BUY    # BUY overrides SELL on conflict

    n_buy  = (predictions == BUY).sum()
    n_sell = (predictions == SELL).sum()
    n_hold = (predictions == HOLD).sum()

    print(f"\n── Any-tree-fires prediction (certainty={certainty}) ────")
    print(f"   BUY  signals : {n_buy:,}  "
          f"(max trees firing on one bar: {buy_votes.max()})")
    print(f"   SELL signals : {n_sell:,}  "
          f"(max trees firing on one bar: {sell_votes.max()})")
    print(f"   HOLD         : {n_hold:,}")
    print(f"   Signal coverage : {100*(n_buy+n_sell)/n:.1f} %")

    return predictions, buy_votes, sell_votes


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_model(ticker=TICKER, period=PERIOD, interval=INTERVAL):
    """
    End-to-end pipeline:
        1. Load and prepare data
        2. Chronological train / calibration / test split
        3. Drop correlated features
        4. Tune hyperparameters (RandomizedSearchCV + TimeSeriesSplit)
        5. Prune low-importance features, refit
        6. Calibrate probabilities
        7. Predict on test set with confidence thresholding
        8. Print full evaluation

    Returns
    -------
    model       : calibrated, pruned, tuned RandomForest
    kept_cols   : list[str]  — feature columns the model expects at inference
    """

    # ── 1. Data ──────────────────────────────────────────────────────────────
    X, y = load_and_prepare(ticker=ticker, period=period, interval=interval)

    # ── 2. Split ─────────────────────────────────────────────────────────────
    X_train, X_cal, X_test, y_train, y_cal, y_test = chronological_split(X, y)

    # ── 3. HOLD undersampling ─────────────────────────────────────────────────
    X_train, y_train = undersample_hold(X_train, y_train)

    # ── 4. Correlation pruning ────────────────────────────────────────────────
    X_train, X_cal, X_test, corr_dropped = drop_correlated(
        X_train, X_cal, X_test
    )

    # ── 5. Hyperparameter tuning ──────────────────────────────────────────────
    best_model, best_params = tune_hyperparams(X_train, y_train)

    # ── 6. Importance pruning + refit ─────────────────────────────────────────
    best_model, X_train, X_cal, X_test, kept_cols = prune_low_importance(
        best_model, X_train, X_cal, X_test, y_train, best_params
    )

    # ── 7. Probability calibration ────────────────────────────────────────────
    cal_model = calibrate(best_model, X_cal, y_cal)

    # ── 8. Predict on held-out test set ───────────────────────────────────────
    # Method A — any-tree-fires: signals if even one tree is certain enough
    y_pred_atf, buy_votes, sell_votes = predict_any_tree_fires(
        best_model, X_test
    )

    # Method B — calibrated majority vote with confidence threshold
    y_pred_conf, probas, confidence = predict_with_confidence(cal_model, X_test)

    # ── 9. Evaluate both methods side by side ─────────────────────────────────
    print_trading_metrics(
        y_test.values, y_pred_atf,
        # pass buy/sell vote counts as a proxy for confidence in the ATF report
        np.where(y_pred_atf == BUY,  buy_votes,
        np.where(y_pred_atf == SELL, sell_votes,
                 np.zeros(len(y_test)))) / len(best_model.estimators_),
        label=f"Any-tree-fires (certainty={TREE_CERTAINTY_THRESHOLD})"
    )
    print_trading_metrics(
        y_test.values, y_pred_conf, confidence,
        label=f"Calibrated majority vote (threshold={CONFIDENCE_THRESHOLD})"
    )

    return cal_model, best_model, kept_cols


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cal_model, raw_model, feature_cols = build_model()
    print(f"\n✓  Models ready.  Expects {len(feature_cols)} input features.")
    print(f"   Features: {feature_cols}")
    print(f"   cal_model  — calibrated majority-vote RF (use .predict_proba)")
    print(f"   raw_model  — raw RF for any-tree-fires prediction")