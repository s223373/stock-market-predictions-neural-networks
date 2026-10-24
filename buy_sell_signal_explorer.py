"""
signal_explorer.py
==================
Takes the BUY/SELL labels produced by build_labeled_dataset() and asks:

  1. What do the signals actually look like afterwards? (MFE / MAE)
  2. Which target % / stop % pair gives the best expectancy?
  3. How long (in bars / minutes) does a trade take to reach its target?

Every trade is simulated bar-by-bar on real High/Low data:
  - Entry  : Close of the signal bar (+ ENTRY_LAG_BARS)
  - Win    : price touches target before stop
  - Loss   : price touches stop first (if both touch in the same bar, the
             stop is assumed to hit first -> conservative)
  - Timeout: neither hit within MAX_HOLD_BARS -> exit at that bar's Close

Params are picked on the first TRAIN_FRAC of signals (chronological) and then
re-tested on the remaining signals so you can see if the "optimum" survives.

Usage
-----
  python signal_explorer.py

Outputs
-------
  signal_explorer_grid_<SIDE>.csv   full target x stop grid
  signal_explorer_heatmap.png       expectancy heatmaps
"""

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from dataset_builder import (
        build_labeled_dataset, dedupe_consecutive_signals,
        TICKER, PERIOD, INTERVAL, BUY, SELL, _build_threshold,
    )
except ImportError:
    from dataset_builder import (
        build_labeled_dataset, dedupe_consecutive_signals,
        TICKER, PERIOD, INTERVAL, BUY, SELL, _build_threshold,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MAX_HOLD_BARS   = 60      # timeout horizon (bars). 60 x 1m = 1 hour
ENTRY_LAG_BARS  = 0       # 0 = fill at signal bar's Close; 1 = fill one bar later
COST_PCT        = 0.10    # ROUND-TRIP fees + slippage, in percent (0.10 = 0.10%)
COLLAPSE_RUNS   = True    # keep only last bar of consecutive same-side signals
TRAIN_FRAC      = 0.60    # fraction of signals used to pick params
MIN_TRADES      = 30      # ignore grid cells with fewer trades than this
BASELINE_STEP   = 3       # sample every Nth bar for the "random entry" baseline

TARGETS = np.round(np.arange(0.05, 1.0001, 0.05), 2)   # percent
STOPS   = np.round(np.arange(0.05, 1.0001, 0.05), 2)   # percent
TTT_LEVELS = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]      # for time-to-target table


def _bar_minutes(interval: str) -> float:
    n = int("".join(ch for ch in interval if ch.isdigit()))
    unit = interval[-1]
    return n * {"m": 1, "h": 60, "d": 1440}.get(unit, 1)


BAR_MIN = _bar_minutes(INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREP
# ─────────────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    raw = yf.download(TICKER, period=PERIOD, interval=INTERVAL,
                      progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def forward_arrays(raw: pd.DataFrame, pos: np.ndarray, long: bool):
    """
    For each entry position build (n x H) matrices of:
        fav  – best-case move in the trade's favour, in %
        adv  – worst-case move against the trade, in %
    and `final` – % return if held to the timeout bar.
    Signals without a full MAX_HOLD_BARS window ahead are dropped.
    """
    H = MAX_HOLD_BARS
    high  = raw["High"].to_numpy(float)
    low   = raw["Low"].to_numpy(float)
    close = raw["Close"].to_numpy(float)

    idx = pos[:, None] + ENTRY_LAG_BARS + np.arange(1, H + 1)[None, :]
    ok  = idx[:, -1] < len(raw)
    pos, idx = pos[ok], idx[ok]

    entry = close[pos + ENTRY_LAG_BARS][:, None]
    hi, lo, last = high[idx], low[idx], close[idx[:, -1]][:, None]

    if long:
        fav   = (hi / entry - 1) * 100
        adv   = (1 - lo / entry) * 100
        final = ((last / entry) - 1)[:, 0] * 100
    else:
        fav   = (1 - lo / entry) * 100
        adv   = (hi / entry - 1) * 100
        final = (1 - last / entry)[:, 0] * 100
    return pos, fav, adv, final


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate(fav, adv, final, tgt, stp):
    """Vectorised bar-by-bar outcome for one (target, stop) pair."""
    n, H = fav.shape
    hit_t, hit_s = fav >= tgt, adv >= stp
    t_idx = np.where(hit_t.any(1), hit_t.argmax(1), H)
    s_idx = np.where(hit_s.any(1), hit_s.argmax(1), H)

    win  = t_idx < s_idx                        # stop wins ties (conservative)
    loss = (s_idx <= t_idx) & (s_idx < H)
    tout = ~win & ~loss

    pnl  = np.where(win, tgt, np.where(loss, -stp, final)) - COST_PCT
    held = np.where(win, t_idx + 1, np.where(loss, s_idx + 1, H))

    gross_w, gross_l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "n":            n,
        "win_%":        100 * win.mean(),
        "loss_%":       100 * loss.mean(),
        "timeout_%":    100 * tout.mean(),
        "expectancy_%": pnl.mean(),
        "total_%":      pnl.sum(),
        "profit_factor": gross_w / gross_l if gross_l > 0 else np.inf,
        "bars_to_target": held[win].mean() if win.any() else np.nan,
        "bars_held":    held.mean(),
    }


def run_grid(fav, adv, final) -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        for s in STOPS:
            r = simulate(fav, adv, final, t, s)
            r.update(target=t, stop=s)
            rows.append(r)
    cols = ["target", "stop"] + [c for c in rows[0] if c not in ("target", "stop")]
    return pd.DataFrame(rows)[cols]


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def excursion_report(fav, adv, side):
    print(f"\n  {side} – how far does price go after the signal "
          f"(next {MAX_HOLD_BARS} bars)?")
    mfe, mae = fav.max(1), adv.max(1)
    q = [0.25, 0.50, 0.75, 0.90]
    print("      quantile        " + "".join(f"{int(x*100):>8d}%" for x in q))
    print("      MFE (best, %)   " + "".join(f"{np.quantile(mfe, x):>9.3f}" for x in q))
    print("      MAE (worst, %)  " + "".join(f"{np.quantile(mae, x):>9.3f}" for x in q))


def time_to_target_report(fav, side):
    print(f"\n  {side} – time to reach a target (ignoring stops)")
    print(f"      {'target':>7} {'reached':>9} {'median bars':>12} "
          f"{'mean bars':>10} {'mean min':>9}")
    for lvl in TTT_LEVELS:
        hit = fav >= lvl
        reached = hit.any(1)
        if not reached.any():
            print(f"      {lvl:>6.2f}% {0:>8.1f}%  {'-':>12} {'-':>10} {'-':>9}")
            continue
        first = hit.argmax(1)[reached] + 1
        print(f"      {lvl:>6.2f}% {100*reached.mean():>8.1f}% "
              f"{np.median(first):>12.1f} {first.mean():>10.1f} "
              f"{first.mean()*BAR_MIN:>9.1f}")


def fmt(df):
    return df.to_string(index=False, float_format=lambda x: f"{x:,.3f}")


def analyse_side(raw, labeled, code, side, long):
    sig = labeled["target"].copy()
    if COLLAPSE_RUNS:
        n_raw = int((sig == code).sum())
        sig = dedupe_consecutive_signals(sig)
        print(f"\n{'═'*70}\n  {side}: {n_raw:,} labeled bars -> "
              f"{int((sig == code).sum()):,} after collapsing runs")
    else:
        print(f"\n{'═'*70}\n  {side}: {int((sig == code).sum()):,} signals")

    sig_idx = sig.index[sig == code]
    pos = raw.index.get_indexer(sig_idx)
    pos = pos[pos >= 0]
    pos, fav, adv, final = forward_arrays(raw, pos, long)
    n = len(pos)
    if n < MIN_TRADES * 2:
        print(f"  Only {n} usable signals – need more data. Skipping.")
        return None
    print(f"  {n:,} signals have a full {MAX_HOLD_BARS}-bar window ahead.")

    excursion_report(fav, adv, side)
    time_to_target_report(fav, side)

    # ── grid search on train, validate on test ──────────────────────────────
    cut = int(n * TRAIN_FRAC)
    grid_all   = run_grid(fav, adv, final)
    grid_train = run_grid(fav[:cut], adv[:cut], final[:cut])
    grid_all.to_csv(f"signal_explorer_grid_{side}.csv", index=False)

    ok = grid_train[grid_train["n"] >= MIN_TRADES]
    top = ok.sort_values("expectancy_%", ascending=False).head(8)
    print(f"\n  {side} – top target/stop pairs on TRAIN "
          f"(first {cut} signals, net of {COST_PCT}% cost)")
    print(fmt(top[["target", "stop", "win_%", "loss_%", "timeout_%",
                   "expectancy_%", "profit_factor", "bars_to_target"]]))

    best = top.iloc[0]
    t, s = best["target"], best["stop"]
    test = simulate(fav[cut:], adv[cut:], final[cut:], t, s)
    full = simulate(fav, adv, final, t, s)

    # baseline: same trade on arbitrary bars (is the label adding anything?)
    all_pos = np.arange(0, len(raw), BASELINE_STEP)
    _, bf, ba, bfin = forward_arrays(raw, all_pos, long)
    base = simulate(bf, ba, bfin, t, s)

    print(f"\n  {side} – best TRAIN pair: target {t:.2f}%  stop {s:.2f}%")
    cmp = pd.DataFrame({
        "train": best[list(test.keys())].astype(float).values,
        "test (out-of-sample)": list(test.values()),
        "all signals": list(full.values()),
        "random-bar baseline": list(base.values()),
    }, index=list(test.keys())).drop(index=["total_%"])
    print(cmp.to_string(float_format=lambda x: f"{x:,.3f}"))

    avg_min = test["bars_to_target"] * BAR_MIN if not np.isnan(test["bars_to_target"]) else np.nan
    print(f"\n  → Winners take ~{full['bars_to_target']:.1f} bars "
          f"(~{full['bars_to_target']*BAR_MIN:.1f} min) to hit {t:.2f}% on average; "
          f"average trade lasts {full['bars_held']:.1f} bars.")
    return grid_all, (t, s)


def plot_heatmaps(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping heatmap.")
        return
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6), squeeze=False)
    for ax, (side, (grid, best)) in zip(axes[0], results.items()):
        piv = grid.pivot(index="stop", columns="target", values="expectancy_%")
        lim = np.nanmax(np.abs(piv.values))
        im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="RdYlGn",
                       vmin=-lim, vmax=lim,
                       extent=[TARGETS[0], TARGETS[-1], STOPS[0], STOPS[-1]])
        ax.plot(best[0], best[1], "k*", markersize=16)
        ax.set_title(f"{side}: expectancy per trade (%)  ★ = best on train")
        ax.set_xlabel("target %")
        ax.set_ylabel("stop %")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("signal_explorer_heatmap.png", dpi=130)
    print("\n✓ Saved signal_explorer_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pd.set_option("display.width", 160)

    raw     = load_raw()
    labeled = build_labeled_dataset(TICKER, PERIOD, INTERVAL)

    # What threshold is the labeler actually using?
    thr = _build_threshold(raw) * 100
    print(f"\nLabeler threshold (% move needed over look-ahead): "
          f"mean {thr.mean():.3f}%  median {thr.median():.3f}%  "
          f"p10 {thr.quantile(.1):.3f}%  p90 {thr.quantile(.9):.3f}%")

    results = {}
    for code, side, long in [(BUY, "BUY", True), (SELL, "SELL", False)]:
        out = analyse_side(raw, labeled, code, side, long)
        if out:
            results[side] = out

    if results:
        plot_heatmaps(results)

    print("\nNOTE: labels are built from FUTURE returns, so these numbers are an "
          "upper bound\n      (what a perfect-foresight model would capture), not "
          "what a live model will get.")