"""
nb_exit_grid_search.py
======================
Grid search over TARGET_PCT x STOP_PCT x MAX_HOLD_BARS, from your current
values toward the shorter/tighter ones, using the same walk-forward machinery
as nb_bunched_backtest.py (drop this file next to it; nothing there changes).

How combos are judged
---------------------
  * SELECTION uses validation slices only: for each combo and each fold the
    best (alpha, threshold) is found on the fold's validation slice, and the
    combo's score = mean of those best validation t-stats across folds.
  * The TEST region is only used to REPORT what each combo would have done.
    Ranking by test return would just pick the luckiest combo, so the table
    is sorted by validation score and the best-by-test row is labelled
    informational.

Usage
-----
    python nb_exit_grid_search.py
Output
------
    nb_exit_grid_results.csv   every combo, sorted by validation score
    nb_exit_grid.png           heatmaps: validation score (top) vs test return (bottom)
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import nb_bunched_backtest as bt

# ── Grid (your values first, then toward the predicted ones) ─────────────────
START     = dict(target=0.0030, stop=0.0030, hold=25)   # your current settings
PREDICTED = dict(target=0.0012, stop=0.0012, hold=20)   # the suggested settings

TARGET_GRID = [0.01, 0.005, 0.0030, 0.0025, 0.0020, 0.0015, 0.0012, 0.0010]
STOP_GRID   = [0.01, 0.005, 0.0030, 0.0025, 0.0020, 0.0015, 0.0012, 0.0010]
HOLD_GRID   = [100, 75, 50, 25, 20, 15, 10]

FEE_BPS = None            # None = use nb_bunched_backtest.FEE_BPS; else e.g. 5.0 per side
TOP_N = 10                # rows printed
RESULTS_CSV = "nb_exit_grid_results.csv"
PLOT_PATH = "nb_exit_grid.png"

EMPTY_EVENTS = ["start_ts", "end_ts", "offset", "pred"]


def _grid(values, *must_include):
    return sorted(set(values) | set(must_include), reverse=True)


TARGET_GRID = _grid(TARGET_GRID, START["target"], PREDICTED["target"])
STOP_GRID = _grid(STOP_GRID, START["stop"], PREDICTED["stop"])
HOLD_GRID = _grid(HOLD_GRID, START["hold"], PREDICTED["hold"])


def set_exits(tp, sp, hold):
    """simulate() reads these module globals at call time."""
    bt.TARGET_PCT, bt.STOP_PCT, bt.MAX_HOLD_BARS = tp, sp, hold
    if FEE_BPS is not None:
        bt.FEE_BPS = FEE_BPS


def cat(parts):
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EMPTY_EVENTS)


# ─────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE EVERYTHING THAT DOESN'T DEPEND ON THE EXIT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

def prepare_folds(datasets, base, bool_cols):
    """Per fold: validation signals for every (alpha, thr), test probabilities for
    every alpha, and the baseline test events. Computed once, reused for all combos."""
    px = base["Close"]
    n = len(px)
    bar = px.index.to_series().diff().median()
    embargo = bt.FORWARD_BARS * bar
    edges = np.linspace(int(n * bt.TRAIN_FRAC), n, bt.N_FOLDS + 1).astype(int)

    def fit_start_for(pos):
        return px.index[max(0, pos - bt.ROLLING_TRAIN_BARS)] if bt.ROLLING_TRAIN_BARS else px.index[0]

    folds = []
    for f in range(bt.N_FOLDS):
        t0 = px.index[edges[f]]
        t1 = px.index[edges[f + 1]] if edges[f + 1] < n else px.index[-1] + bar
        val_pos = int(edges[f] * (1 - bt.VAL_FRAC))
        val_start = px.index[val_pos]
        px_val = px[(px.index >= val_start) & (px.index < t0)]

        test_scored, val_sigs = {}, {}
        for alpha in bt.ALPHA_GRID:
            m_val = bt.fit_models(datasets, bool_cols, alpha, fit_start_for(val_pos), val_start - embargo)
            m_test = bt.fit_models(datasets, bool_cols, alpha, fit_start_for(edges[f]), t0 - embargo)
            if m_val is None or m_test is None:
                continue
            scored_val = bt.score_blocks(m_val, datasets, bool_cols, val_start, t0)
            test_scored[alpha] = bt.score_blocks(m_test, datasets, bool_cols, t0, t1)
            for thr in bt.THRESH_GRID:
                events = bt.to_events(scored_val, thr)
                val_sigs[(alpha, thr)] = bt.prepare_signals(events, px_val.index)[1]

        m_base = bt.fit_models(datasets, bool_cols, bt.BASELINE_ALPHA, fit_start_for(edges[f]), t0 - embargo)
        ev_base = (bt.to_events(bt.score_blocks(m_base, datasets, bool_cols, t0, t1), None)
                   if m_base is not None else pd.DataFrame(columns=EMPTY_EVENTS))

        folds.append({"px_val": px_val.values, "val_sigs": val_sigs,
                      "test_scored": test_scored, "ev_base": ev_base})

    cutoff = px.index[edges[0]]
    return folds, cutoff


def score_signals(close, signals_at):
    """Same objective as nb_bunched_backtest.validation_score, on cached signals."""
    _, trades, _, _ = bt.simulate(close, signals_at)
    r = np.asarray(trades)
    if len(r) < bt.MIN_VAL_TRADES:
        return -np.inf, len(r)
    sd = max(r.std(ddof=1), 1e-6)
    return float(r.mean() / sd * np.sqrt(len(r))), len(r)


def run_stats(equity, trades, exits):
    t = np.asarray(trades)
    return {
        "return": equity[-1] - 1,
        "trades": len(t),
        "win": float((t > 0).mean()) if len(t) else np.nan,
        "maxdd": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "timeout_frac": exits["timeout"] / len(t) if len(t) else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────────────────────────────

def run_grid():
    datasets, base = bt.build_bunched_datasets(return_base=True)
    bool_cols = [c for c in datasets[0].columns if c not in bt.META_COLS]

    print("\nPre-computing fold models (independent of exit settings) ...")
    folds, cutoff = prepare_folds(datasets, base, bool_cols)

    px = base["Close"]
    px_test = px[px.index >= cutoff]
    close = px_test.values

    base_signals = bt.prepare_signals(cat([f["ev_base"] for f in folds]), px_test.index)[1]

    # How often could each target/stop even be reached within the hold?
    reach = {}
    s = pd.Series(close)
    for h in HOLD_GRID:
        up = (s[::-1].rolling(h).max()[::-1].shift(-1) / s - 1).dropna()
        dn = (1 - s[::-1].rolling(h).min()[::-1].shift(-1) / s).dropna()
        reach[h] = (up.values, dn.values)

    combos = [(tp, sp, h) for h in HOLD_GRID for tp in TARGET_GRID for sp in STOP_GRID]
    print(f"Running {len(combos)} exit combinations "
          f"({len(TARGET_GRID)} targets x {len(STOP_GRID)} stops x {len(HOLD_GRID)} holds) ...")

    cache, rows = {}, []
    t_start = time.time()
    for ci, (tp, sp, hold) in enumerate(combos, 1):
        set_exits(tp, sp, hold)

        # ── select (alpha, thr) per fold on VALIDATION only ──────────────────
        choices, fold_scores, val_trades = [], [], 0
        for fd in folds:
            best = (-np.inf, None, None, 0)
            for (alpha, thr), sig in fd["val_sigs"].items():
                score, n_tr = score_signals(fd["px_val"], sig)
                if score > best[0]:
                    best = (score, alpha, thr, n_tr)
            score, alpha, thr, n_tr = best
            if np.isfinite(score):
                choices.append((alpha, thr))
                val_trades += n_tr
            else:
                choices.append(("base", None))              # fallback = baseline settings
            fold_scores.append(score)

        sel_score = float(np.mean(fold_scores)) if np.all(np.isfinite(fold_scores)) else -np.inf

        # ── report TEST result of those choices ───────────────────────────────
        key = tuple(choices)
        if key not in cache:
            parts = [fd["ev_base"] if a == "base" else bt.to_events(fd["test_scored"][a], thr)
                     for fd, (a, thr) in zip(folds, choices)]
            cache[key] = bt.prepare_signals(cat(parts), px_test.index)[1]
        eq, tr, _, ex = bt.simulate(close, cache[key])
        tuned = run_stats(eq, tr, ex)
        eq_b, tr_b, _, ex_b = bt.simulate(close, base_signals)
        basel = run_stats(eq_b, tr_b, ex_b)

        up, dn = reach[hold]
        tag = ("START" if (tp, sp, hold) == (START["target"], START["stop"], START["hold"])
               else "PREDICTED" if (tp, sp, hold) == (PREDICTED["target"], PREDICTED["stop"], PREDICTED["hold"])
               else "")
        rows.append({
            "tag": tag, "target": tp, "stop": sp, "hold": hold,
            "val_score": sel_score, "val_trades": val_trades,
            "test_return": tuned["return"], "test_trades": tuned["trades"],
            "test_win": tuned["win"], "test_maxdd": tuned["maxdd"],
            "timeout_frac": tuned["timeout_frac"],
            "base_return": basel["return"], "base_trades": basel["trades"],
            "tgt_reach": float((up >= tp).mean()), "stop_reach": float((dn >= sp).mean()),
            "folds_fallback": int(sum(c[0] == "base" for c in choices)),
        })
        if ci % 20 == 0 or ci == len(combos):
            print(f"  {ci}/{len(combos)} done  ({time.time() - t_start:.0f}s)")

    res = pd.DataFrame(rows)
    res["_k"] = res["val_score"].replace(-np.inf, -1e9)
    res = res.sort_values(["_k", "test_return"], ascending=False).drop(columns="_k").reset_index(drop=True)
    res.to_csv(RESULTS_CSV, index=False)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(df):
    out = df.copy()
    out["target"] = out["target"].map("{:.2%}".format)
    out["stop"] = out["stop"].map("{:.2%}".format)
    for c in ("test_return", "test_maxdd", "base_return"):
        out[c] = out[c].map("{:.2%}".format)
    for c in ("test_win", "timeout_frac", "tgt_reach", "stop_reach"):
        out[c] = out[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.0%}")
    out["val_score"] = out["val_score"].map(lambda v: "  -inf" if not np.isfinite(v) else f"{v:6.2f}")
    return out


def report(res):
    cols = ["tag", "target", "stop", "hold", "val_score", "val_trades", "test_return",
            "test_trades", "test_win", "timeout_frac", "tgt_reach", "stop_reach", "base_return"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print(f"\n{'═' * 100}\n  TOP {TOP_N} by VALIDATION score (selection metric)\n{'═' * 100}")
    print(_fmt(res.head(TOP_N))[cols].to_string(index=False))

    for name in ("START", "PREDICTED"):
        row = res[res["tag"] == name]
        if len(row) and row.index[0] >= TOP_N:
            rank = row.index[0] + 1
            print(f"\n  {name} settings (rank {rank} of {len(res)}):")
            print(_fmt(row)[cols].to_string(index=False))

    best_test = res.sort_values("test_return", ascending=False).head(3)
    print(f"\n  Best 3 by TEST return (informational only - not used for selection):")
    print(_fmt(best_test)[cols].to_string(index=False))

    ok = res[np.isfinite(res["val_score"])]
    if len(ok) >= 5:
        corr = ok["val_score"].corr(ok["test_return"], method="spearman")
        print(f"\n  Rank correlation, validation score vs test return across combos: {corr:+.2f}")
        print("  (near 0 or negative => validation isn't predicting test; treat any winner with suspicion)")


def plot_heatmaps(res):
    fig, axes = plt.subplots(2, len(HOLD_GRID), figsize=(4.4 * len(HOLD_GRID), 8.5), squeeze=False)
    t_lab = [f"{t:.2%}" for t in TARGET_GRID]
    s_lab = [f"{s:.2%}" for s in STOP_GRID]

    for col, hold in enumerate(HOLD_GRID):
        sub = res[res["hold"] == hold]
        for row, (metric, title) in enumerate((("val_score", "validation score"),
                                               ("test_return", "test return"))):
            M = np.full((len(STOP_GRID), len(TARGET_GRID)), np.nan)
            for _, r in sub.iterrows():
                v = r[metric]
                if np.isfinite(v):
                    M[STOP_GRID.index(r["stop"]), TARGET_GRID.index(r["target"])] = v
            vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
            ax = axes[row][col]
            im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(TARGET_GRID)))
            ax.set_xticklabels(t_lab, rotation=45, fontsize=7)
            ax.set_yticks(range(len(STOP_GRID)))
            ax.set_yticklabels(s_lab, fontsize=7)
            ax.set_xlabel("target", fontsize=8)
            ax.set_ylabel("stop", fontsize=8)
            ax.set_title(f"hold={hold}  |  {title}", fontsize=9)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        txt = f"{M[i, j]:.1f}" if metric == "val_score" else f"{M[i, j]:.1%}"
                        ax.text(j, i, txt, ha="center", va="center", fontsize=6)
            for name, spec, color in (("START", START, "blue"), ("PREDICTED", PREDICTED, "magenta")):
                if spec["hold"] == hold:
                    ax.add_patch(Rectangle((TARGET_GRID.index(spec["target"]) - 0.5,
                                            STOP_GRID.index(spec["stop"]) - 0.5), 1, 1,
                                           fill=False, edgecolor=color, linewidth=2))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Exit grid search  (blue box = your START settings, magenta = PREDICTED settings)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=140)
    plt.show()


if __name__ == "__main__":
    results = run_grid()
    report(results)
    plot_heatmaps(results)
    print(f"\nSaved → {RESULTS_CSV}, {PLOT_PATH}")
    print("Copy the settings you choose into MAX_HOLD_BARS / TARGET_PCT / STOP_PCT "
          "in nb_bunched_backtest.py.")