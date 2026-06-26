"""
rf_backtest.py
==============
Walk-Forward Backtest for the Random Forest BUY / HOLD / SELL classifier.

Mirrors the structure and output of lstm_backtest.py, with two prediction
modes run side-by-side so you can directly compare them.

Prediction modes
----------------
any_tree_fires  — signals BUY/SELL if any single tree fires with sufficient
                  leaf purity. Higher recall, lower precision.
confidence      — calibrated majority-vote; signals only when predicted
                  probability exceeds CONFIDENCE_THRESHOLD. Lower recall,
                  higher precision.

How actual outcomes are determined
-----------------------------------
For each bar t, the model predicts BUY/HOLD/SELL.  The actual outcome is
determined by looking at the real Close price FORWARD_BARS bars later:

    actual_pct = (close[t + FORWARD_BARS] - close[t]) / close[t] * 100

    actual_pct >  PCT_THRESHOLD  →  actual = BUY
    actual_pct < -PCT_THRESHOLD  →  actual = SELL
    otherwise                    →  actual = HOLD

This matches how the labels were generated in build_labeled_dataset.py,
except we use a fixed PCT_THRESHOLD here (the same as the LSTM backtest)
rather than the ATR-adaptive threshold used during training.  This gives
a fairer apples-to-apples comparison with the LSTM backtest results.

Usage
-----
    python rf_backtest.py
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

from build_labeled_dataset import build_labeled_dataset
from random_forest_model import (
    # Pipeline functions
    load_and_prepare,
    chronological_split,
    drop_correlated,
    tune_hyperparams,
    prune_low_importance,
    calibrate,
    predict_any_tree_fires,
    predict_with_confidence,
    undersample_hold,
    # Constants — must match random_forest_model.py exactly
    TRAIN_FRAC,
    CAL_FRAC,
    BUY,
    HOLD,
    SELL,
    TREE_CERTAINTY_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    TICKER,
    PERIOD,
    INTERVAL,
    FORWARD_BARS,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# How far apart predictions are made.  1 = every bar; 5 = every 5th bar.
# Matches the LSTM backtest STEP parameter.  Lower = more trades, slower P&L
# loop (though still fast since all predictions are batched before the loop).
STEP = 1

# Fixed percentage threshold for determining actual BUY/SELL/HOLD outcome
# from real forward price movement.  Mirrors lstm_backtest.THRESHOLD.
PCT_THRESHOLD = 0.3   # ±0.3 %

# Round-trip commission (applied once for entry + once for exit)
COMMISSION = 0.001    # 0.1 % per side = 0.2 % round-trip

SAVE_RESULTS = "rf_backtest_results.csv"

# Label strings (for display only)
LABEL = {BUY: "BUY", HOLD: "HOLD", SELL: "SELL"}


# ─────────────────────────────────────────────────────────────────────────────
# DATA + MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_data_and_model():
    """
    Build the labeled dataset, train the full RF pipeline, and return:

        cal_model    — CalibratedClassifierCV for confidence-threshold mode
        raw_model    — raw RandomForestClassifier for any-tree-fires mode
        X_test       — feature matrix for the held-out test period
        close_test   — raw Close prices aligned to X_test rows
        dates_test   — datetime index aligned to X_test rows
        feature_cols — ordered list of feature columns the model expects

    Why not just call build_model() from random_forest_model.py?
    -------------------------------------------------------------
    build_model() trains the model but doesn't return the test split or the
    raw Close prices needed to compute actual forward P&L.  We replicate the
    same pipeline steps here so we can intercept the test slice before Close
    is dropped from the feature matrix.
    """

    # ── 1. Build labeled dataset ──────────────────────────────────────────────
    # labeled has: Close + all boolean features + target, datetime index
    print("\n[backtest] Building dataset and training model …")
    labeled = build_labeled_dataset(
        ticker=TICKER, period=PERIOD, interval=INTERVAL
    )
    full_df = labeled.reset_index()   # brings Datetime into a column

    # ── 2. Compute chronological test-split boundaries ────────────────────────
    n      = len(full_df)
    i_cal  = int(n * TRAIN_FRAC)
    i_test = int(n * (TRAIN_FRAC + CAL_FRAC))

    # Save Close prices and dates for the test slice NOW, before Close is
    # dropped from the feature matrix.  These are needed to compute actual
    # forward returns and P&L in the backtest loop.
    test_slice  = full_df.iloc[i_test:].copy()
    dates_test  = pd.to_datetime(test_slice["Datetime"].values)
    close_test  = test_slice["Close"].values.astype(float)

    # ── 3. Build feature matrix (mirrors load_and_prepare) ───────────────────
    drop = [c for c in ["Datetime", "Close", "target"] if c in full_df.columns]
    X    = full_df.drop(columns=drop)
    y    = full_df["target"]

    # ── 4. Chronological split ────────────────────────────────────────────────
    X_train, y_train = X.iloc[:i_cal],        y.iloc[:i_cal]
    X_cal,   y_cal   = X.iloc[i_cal:i_test],  y.iloc[i_cal:i_test]
    X_test,  y_test  = X.iloc[i_test:],       y.iloc[i_test:]

    print(f"[backtest] Test set: {len(X_test):,} bars  "
          f"({dates_test[0].strftime('%Y-%m-%d')} → "
          f"{dates_test[-1].strftime('%Y-%m-%d')})")

    # ── 5. Run the same training pipeline as random_forest_model.py ──────────
    X_train, y_train                           = undersample_hold(X_train, y_train)
    X_train, X_cal, X_test, _                  = drop_correlated(X_train, X_cal, X_test)
    raw_model, best_params                     = tune_hyperparams(X_train, y_train)
    raw_model, X_train, X_cal, X_test, f_cols = prune_low_importance(
        raw_model, X_train, X_cal, X_test, y_train, best_params
    )
    cal_model = calibrate(raw_model, X_cal, y_cal)

    return cal_model, raw_model, X_test, y_test, close_test, dates_test, f_cols


# ─────────────────────────────────────────────────────────────────────────────
# ACTUAL OUTCOME
# ─────────────────────────────────────────────────────────────────────────────

def actual_outcome(close_test: np.ndarray, i: int) -> tuple[str, float]:
    """
    Determine the true BUY/HOLD/SELL outcome for bar i by looking at the
    actual Close price FORWARD_BARS bars later.

    Returns
    -------
    outcome : "BUY" | "HOLD" | "SELL"
    pct     : actual percentage return over the forward window
    """
    entry = close_test[i]
    exit_ = close_test[i + FORWARD_BARS]
    pct   = (exit_ - entry) / (abs(entry) + 1e-9) * 100

    if   pct >  PCT_THRESHOLD: return "BUY",  pct
    elif pct < -PCT_THRESHOLD: return "SELL", pct
    else:                      return "HOLD", pct


def pnl_for_signal(signal: str, entry: float, exit_: float) -> float:
    """Compute round-trip P&L percentage including commission."""
    if signal == "BUY":
        return ((exit_ - entry) / entry - 2 * COMMISSION) * 100
    if signal == "SELL":
        return ((entry - exit_) / entry - 2 * COMMISSION) * 100
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    raw_model,
    cal_model,
    X_test:      pd.DataFrame,
    close_test:  np.ndarray,
    dates_test:  np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Batch-predict across the full test set, then loop through bars to compute
    P&L from actual prices.  Returns two result DataFrames — one per mode.

    Why batch before the loop?
    --------------------------
    predict_any_tree_fires() iterates all trees in the forest for every sample.
    Calling it inside a bar-by-bar loop would repeat that work n_test times.
    Predicting the full test set in one call amortises the tree-iteration cost
    across all samples at once — sklearn's tree traversal is vectorised in C,
    so batch prediction is orders of magnitude faster than repeated single-row
    calls.
    """
    n = len(X_test)

    # ── Batch predict (both modes) ────────────────────────────────────────────
    print("\n[backtest] Running batch predictions …")

    preds_atf, buy_votes, sell_votes = predict_any_tree_fires(raw_model, X_test)
    preds_conf, probas, confidence   = predict_with_confidence(cal_model, X_test)

    signal_atf  = np.array([LABEL[p] for p in preds_atf])
    signal_conf = np.array([LABEL[p] for p in preds_conf])

    # ── Bar-by-bar P&L loop ───────────────────────────────────────────────────
    rows_atf  = []
    rows_conf = []
    max_i     = n - FORWARD_BARS   # last bar that has a forward window

    total = len(range(0, max_i, STEP))
    print(f"[backtest] {total} windows  |  FORWARD_BARS={FORWARD_BARS}  "
          f"|  STEP={STEP}  |  PCT_THRESHOLD=±{PCT_THRESHOLD}%")

    for idx, i in enumerate(range(0, max_i, STEP)):

        entry_price = close_test[i]
        exit_price  = close_test[i + FORWARD_BARS]
        outcome, actual_pct = actual_outcome(close_test, i)
        entry_date  = dates_test[i]
        exit_date   = dates_test[i + FORWARD_BARS]

        # ── Any-tree-fires ────────────────────────────────────────────────────
        sig_a = signal_atf[i]
        rows_atf.append({
            "entry_date"    : entry_date,
            "exit_date"     : exit_date,
            "entry_price"   : round(float(entry_price), 4),
            "exit_price"    : round(float(exit_price),  4),
            "signal"        : sig_a,
            "actual_outcome": outcome,
            "actual_pct"    : round(actual_pct, 3),
            "correct"       : sig_a == outcome,
            "pnl_pct"       : round(pnl_for_signal(sig_a, entry_price, exit_price), 4),
            "buy_votes"     : int(buy_votes[i]),
            "sell_votes"    : int(sell_votes[i]),
        })

        # ── Confidence threshold ───────────────────────────────────────────────
        sig_c = signal_conf[i]
        rows_conf.append({
            "entry_date"    : entry_date,
            "exit_date"     : exit_date,
            "entry_price"   : round(float(entry_price), 4),
            "exit_price"    : round(float(exit_price),  4),
            "signal"        : sig_c,
            "actual_outcome": outcome,
            "actual_pct"    : round(actual_pct, 3),
            "correct"       : sig_c == outcome,
            "pnl_pct"       : round(pnl_for_signal(sig_c, entry_price, exit_price), 4),
            "max_proba"     : round(float(confidence[i]), 4),
        })

        if (idx + 1) % 100 == 0:
            print(f"  … {idx + 1}/{total} windows done")

    print(f"[backtest] Complete — {len(rows_atf)} windows.\n")
    return pd.DataFrame(rows_atf), pd.DataFrame(rows_conf)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS  (mirrors lstm_backtest.compute_and_print_metrics)
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_print_metrics(results: pd.DataFrame, mode_label: str) -> dict:
    """
    Print a performance report identical in structure to the LSTM backtest:
    signal distribution, accuracy, P&L metrics, equity curve stats,
    confusion matrix, and classification report.
    """
    w = 62

    if len(results) == 0:
        print(f"[{mode_label}] No results.")
        return {}

    predicted  = results                              # RF predicts every bar
    traded     = results[results["signal"] != "HOLD"]
    hold_rows  = results[results["signal"] == "HOLD"]
    buy_rows   = results[results["signal"] == "BUY"]
    sell_rows  = results[results["signal"] == "SELL"]

    sig_counts = results["signal"].value_counts()
    act_counts = results["actual_outcome"].value_counts()

    overall_acc  = (results["signal"] == results["actual_outcome"]).mean() * 100
    dir_acc      = traded["correct"].mean() * 100        if len(traded)     else 0.0
    buy_acc      = buy_rows["correct"].mean()  * 100     if len(buy_rows)   else float("nan")
    sell_acc     = sell_rows["correct"].mean() * 100     if len(sell_rows)  else float("nan")
    hold_acc     = hold_rows["correct"].mean() * 100     if len(hold_rows)  else float("nan")

    winners  = traded[traded["pnl_pct"] > 0]
    losers   = traded[traded["pnl_pct"] <= 0]
    win_rate = len(winners) / len(traded) * 100          if len(traded)  else 0.0
    avg_win  = winners["pnl_pct"].mean()                 if len(winners) else 0.0
    avg_loss = losers["pnl_pct"].mean()                  if len(losers)  else 0.0
    pf       = (winners["pnl_pct"].sum()
                / (abs(losers["pnl_pct"].sum()) + 1e-9)) if len(losers)  else float("inf")
    avg_pnl  = traded["pnl_pct"].mean()                  if len(traded)  else 0.0
    total_pnl= traded["pnl_pct"].sum()

    # Equity curve & drawdown
    equity       = 100.0
    equity_curve = [100.0]
    for pnl in traded["pnl_pct"]:
        equity *= (1 + pnl / 100)
        equity_curve.append(equity)
    cum_return  = equity - 100.0
    equity_arr  = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    max_dd      = ((equity_arr - running_max) / (running_max + 1e-9) * 100).min()

    y_true   = results["actual_outcome"].values
    y_pred   = results["signal"].values
    present  = sorted(set(y_true) | set(y_pred))

    print("\n" + "═" * w)
    print(f"  RF BACKTEST — {mode_label.upper()}")
    print("═" * w)
    print(f"\n  Dataset  : {TICKER} · {PERIOD} · {INTERVAL} bars")
    print(f"  Windows  : {len(results)}  |  Forward: {FORWARD_BARS} bars  "
          f"|  Step: {STEP}")
    print(f"  Threshold: ±{PCT_THRESHOLD}%  |  Commission: {COMMISSION*100:.2f}%/side")

    print(f"\n  ── Signal Distribution {'─' * (w - 26)}")
    for lbl in ["BUY", "HOLD", "SELL"]:
        print(f"    {lbl:<6}  predicted: {sig_counts.get(lbl, 0):>5}   "
              f"actual: {act_counts.get(lbl, 0):>5}")

    print(f"\n  ── Accuracy {'─' * (w - 15)}")
    print(f"    Overall (3-class)      : {overall_acc:>6.1f} %")
    print(f"    Directional (no HOLD)  : {dir_acc:>6.1f} %")
    print(f"    BUY  accuracy          : {buy_acc:>6.1f} %")
    print(f"    SELL accuracy          : {sell_acc:>6.1f} %")
    print(f"    HOLD accuracy          : {hold_acc:>6.1f} %")

    print(f"\n  ── P&L ({COMMISSION*100:.1f}% commission/side) {'─' * (w - 28)}")
    print(f"    Trades taken      : {len(traded):>6}")
    print(f"    Win rate          : {win_rate:>6.1f} %")
    print(f"    Avg win           : {avg_win:>+7.3f} %")
    print(f"    Avg loss          : {avg_loss:>+7.3f} %")
    print(f"    Profit factor     : {pf:>8.3f}")
    print(f"    Avg P&L per trade : {avg_pnl:>+7.3f} %")
    print(f"    Total P&L (sum)   : {total_pnl:>+7.2f} %")
    print(f"    Cumulative return : {cum_return:>+7.2f} %")
    print(f"    Max drawdown      : {max_dd:>+7.2f} %")

    print(f"\n  ── Confusion Matrix {'─' * (w - 23)}")
    cm     = confusion_matrix(y_true, y_pred, labels=present)
    header = "               " + "  ".join(f"Pred {l:<5}" for l in present)
    print(f"    {header}")
    for i, lbl in enumerate(present):
        print(f"    True {lbl:<8}  " + "  ".join(f"{v:>9}" for v in cm[i]))

    print(f"\n  ── Classification Report {'─' * (w - 28)}")
    for line in classification_report(
        y_true, y_pred, labels=present, zero_division=0
    ).split("\n"):
        print(f"    {line}")

    print("═" * w + "\n")

    return dict(
        mode             = mode_label,
        total_windows    = len(results),
        n_trades         = len(traded),
        overall_acc      = overall_acc,
        directional_acc  = dir_acc,
        buy_acc          = buy_acc,
        sell_acc         = sell_acc,
        hold_acc         = hold_acc,
        win_rate         = win_rate,
        avg_win_pct      = avg_win,
        avg_loss_pct     = avg_loss,
        profit_factor    = pf,
        avg_pnl_pct      = avg_pnl,
        total_pnl_pct    = total_pnl,
        cum_return_pct   = cum_return,
        max_drawdown_pct = max_dd,
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(m_atf: dict, m_conf: dict) -> None:
    """
    Print a side-by-side summary of both prediction modes so the trade-off
    between recall (any-tree-fires) and precision (confidence-threshold) is
    immediately visible.
    """
    w = 62
    metrics = [
        ("Trades taken",     "n_trades",         "{:>6.0f}",   "{:>6.0f}"),
        ("Win rate",         "win_rate",          "{:>6.1f} %", "{:>6.1f} %"),
        ("Directional acc",  "directional_acc",   "{:>6.1f} %", "{:>6.1f} %"),
        ("BUY accuracy",     "buy_acc",           "{:>6.1f} %", "{:>6.1f} %"),
        ("SELL accuracy",    "sell_acc",          "{:>6.1f} %", "{:>6.1f} %"),
        ("Profit factor",    "profit_factor",     "{:>8.3f}",   "{:>8.3f}"),
        ("Cumulative return","cum_return_pct",    "{:>+7.2f} %","{:>+7.2f} %"),
        ("Max drawdown",     "max_drawdown_pct",  "{:>+7.2f} %","{:>+7.2f} %"),
    ]

    print("═" * w)
    print("  COMPARISON: Any-Tree-Fires  vs  Confidence-Threshold")
    print("═" * w)
    print(f"  {'Metric':<22}  {'AnyTree':>12}  {'Confidence':>12}")
    print(f"  {'─'*22}  {'─'*12}  {'─'*12}")
    for label, key, fmt_a, fmt_c in metrics:
        va = m_atf.get(key,  float("nan"))
        vc = m_conf.get(key, float("nan"))
        try:
            a_str = fmt_a.format(va)
            c_str = fmt_c.format(vc)
        except (ValueError, TypeError):
            a_str = f"{va:>12}"
            c_str = f"{vc:>12}"
        print(f"  {label:<22}  {a_str:>12}  {c_str:>12}")
    print("═" * w + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Train ─────────────────────────────────────────────────────────────────
    cal_model, raw_model, X_test, y_test, close_test, dates_test, feature_cols = (
        build_data_and_model()
    )

    print(f"\n[backtest] Model trained on {len(feature_cols)} features.")
    print(f"[backtest] Test period  : {dates_test[0].strftime('%Y-%m-%d %H:%M')}"
          f" → {dates_test[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"[backtest] Test bars    : {len(X_test):,}")

    # ── Backtest (both modes in one pass) ─────────────────────────────────────
    results_atf, results_conf = run_backtest(
        raw_model, cal_model, X_test, close_test, dates_test
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    atf_label  = f"Any-Tree-Fires (certainty≥{TREE_CERTAINTY_THRESHOLD})"
    conf_label = f"Confidence-Threshold (≥{CONFIDENCE_THRESHOLD})"

    m_atf  = compute_and_print_metrics(results_atf,  atf_label)
    m_conf = compute_and_print_metrics(results_conf, conf_label)

    print_comparison(m_atf, m_conf)

    # ── Save ──────────────────────────────────────────────────────────────────
    results_atf["mode"]  = "any_tree_fires"
    results_conf["mode"] = "confidence"
    combined = pd.concat([results_atf, results_conf], ignore_index=True)
    combined.to_csv(SAVE_RESULTS, index=False)
    print(f"[output] Results saved → {SAVE_RESULTS}")

    return results_atf, results_conf, m_atf, m_conf


if __name__ == "__main__":
    main()
