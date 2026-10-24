import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB

from dataset_builder import BUY, HOLD, SELL, FORWARD_BARS
from bunched_datasets import build_bunched_datasets

# ── Trade settings ───────────────────────────────────────────────────────────
MAX_HOLD_BARS = 60          # matches the explorer's horizon the pairs were optimized on
FEE_BPS = 1.0

# Side-specific exits, optimized in signal_explorer (best TRAIN pair per side)
BUY_TARGET_PCT  = 0.0040    # +0.40%
BUY_STOP_PCT    = 0.0015    # -0.15%
SELL_TARGET_PCT = 0.0055    # -0.55%
SELL_STOP_PCT   = 0.0035 
TRAIN_FRAC = 0.8             # last (1 - TRAIN_FRAC) of the timeline = walk-forward test region

# "start" -> enter at the FIRST 1m bar of the 5-bar sample (lookahead, optimistic)
# "end"   -> enter at the LAST 1m bar, i.e. when the sample is actually complete
ENTRY_MODE = "end"

# ── Walk-forward + tuning config ─────────────────────────────────────────────
N_FOLDS = 4                  # test region is split into this many consecutive chunks
VAL_FRAC = 0.2               # last 20% of the pre-test history is held out to tune on
ROLLING_TRAIN_BARS = None    # None = expanding window; or e.g. 20000 1m bars = rolling
ALPHA_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]               # NB smoothing
THRESH_GRID = [None, 0.30, 0.35, 0.40, 0.50, 0.60]   # None = plain argmax; else P(class) >= thr
MIN_VAL_TRADES = 10          # a candidate with fewer validation trades is rejected
MIN_FIT_SAMPLES = 100
BASELINE_ALPHA = 1.0         # untuned comparison run (alpha=1, plain argmax)

# ── Plot / output config ─────────────────────────────────────────────────────
PLOT_START = None            # e.g. "2026-10-08 09:30"  (None = start of test period)
PLOT_END = None              # e.g. "2026-10-08 16:00"  (None = end of test period)
DRAW_LEVELS = False          # draw each trade's target/stop lines
SHOW_SKIPPED = True          # faint markers for signals that fired but weren't traded
TRADES_CSV = "nb_bunched_trades.csv"
PARAMS_CSV = "nb_walkforward_params.csv"
PLOT_PATH = "nb_bunched_trades.png"

META_COLS = ["Close", "target", "start_ts"]
SCORE_COLS = ["start_ts", "end_ts", "offset", "p_sell", "p_hold", "p_buy"]


# ─────────────────────────────────────────────────────────────────────────────
# MODELS + SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def fit_models(datasets, bool_cols, alpha, fit_start, fit_end):
    """One BernoulliNB per offset, trained on blocks with
    fit_start <= start_ts and end_ts < fit_end. Returns None if too little data."""
    models = {}
    for k, ds in datasets.items():
        start_ts = pd.DatetimeIndex(ds["start_ts"])
        m = np.asarray((start_ts >= fit_start) & (ds.index < fit_end))
        if m.sum() < MIN_FIT_SAMPLES:
            return None
        model = BernoulliNB(alpha=alpha, binarize=None)   # features are already 0/1
        model.fit(ds.loc[m, bool_cols], ds.loc[m, "target"])
        models[k] = model
    return models


def score_blocks(models, datasets, bool_cols, lo, hi):
    """P(SELL), P(HOLD), P(BUY) for every block (all offsets) with lo <= start_ts < hi."""
    parts = []
    for k, ds in datasets.items():
        start_ts = pd.DatetimeIndex(ds["start_ts"])
        sel = np.asarray((start_ts >= lo) & (start_ts < hi))
        if not sel.any():
            continue
        m = models[k]
        p = np.zeros((int(sel.sum()), 3))                 # columns = codes 0,1,2
        p[:, m.classes_.astype(int)] = m.predict_proba(ds.loc[sel, bool_cols])
        parts.append(pd.DataFrame({
            "start_ts": start_ts[sel], "end_ts": ds.index[sel], "offset": k,
            "p_sell": p[:, 0], "p_hold": p[:, 1], "p_buy": p[:, 2],
        }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SCORE_COLS)


def to_events(scored, thr):
    """Turn probabilities into BUY/SELL events. thr=None -> plain argmax."""
    if len(scored) == 0:
        return pd.DataFrame(columns=["start_ts", "end_ts", "offset", "pred"])
    if thr is None:
        pred = np.argmax(scored[["p_sell", "p_hold", "p_buy"]].values, axis=1)  # 0/1/2 = SELL/HOLD/BUY
    else:
        pred = np.full(len(scored), HOLD)
        buy = ((scored["p_buy"] >= thr) & (scored["p_buy"] > scored["p_sell"])).values
        sell = ((scored["p_sell"] >= thr) & (scored["p_sell"] > scored["p_buy"])).values
        pred[buy] = BUY
        pred[sell] = SELL
    ev = scored[["start_ts", "end_ts", "offset"]].copy()
    ev["pred"] = pred
    return ev[ev["pred"] != HOLD]


def prepare_signals(events, px_index):
    """Map events to the 1m bar where we'd enter. Same-bar BUY/SELL disagreement is skipped."""
    if len(events) == 0:
        return events.assign(i=pd.Series(dtype=int)), {}, {}
    ev = events.copy()
    entry_col = "start_ts" if ENTRY_MODE == "start" else "end_ts"
    ev["i"] = px_index.get_indexer(ev[entry_col])
    ev = ev[ev["i"] >= 0].copy()
    if len(ev) == 0:
        return ev, {}, {}
    g = ev.groupby("i")["pred"]
    signals_at = g.first()[g.nunique() == 1].to_dict()
    offsets_at = ev.groupby("i")["offset"].apply(lambda s: sorted(set(s))).to_dict()
    return ev, signals_at, offsets_at


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate(close, signals_at, offsets_at=None):
    """Bar-by-bar trade simulation. Returns (equity, trades, trade_log, exit_reasons)."""
    offsets_at = offsets_at or {}
    n = len(close)
    equity = np.ones(n + 1)
    position = 0
    entry_price = target = stop = None
    entry_bar = 0
    trades, trade_log, open_trade = [], [], None
    exit_reasons = {"target": 0, "stop": 0, "timeout": 0}

    for i in range(n):
        px = close[i]

        if position != 0:
            hit_target = (position == 1 and px >= target) or (position == -1 and px <= target)
            hit_stop = (position == 1 and px <= stop) or (position == -1 and px >= stop)
            timed_out = (i - entry_bar) >= MAX_HOLD_BARS

            if hit_target or hit_stop or timed_out:
                exit_px = target if hit_target else (stop if hit_stop else px)
                ret = (exit_px - entry_price) / entry_price * position
                fee = FEE_BPS / 10000 * 2
                trades.append(ret - fee)
                reason = "target" if hit_target else "stop" if hit_stop else "timeout"
                exit_reasons[reason] += 1
                trade_log.append({**open_trade, "exit_i": i, "exit_px": exit_px,
                                  "reason": reason, "ret": ret - fee})
                equity[i + 1] = equity[i] * (1 + ret - fee)
                position = 0
            else:
                equity[i + 1] = equity[i]
            continue

        s = signals_at.get(i)
        if s == BUY:
            position = 1
            entry_price = px
            target = px * (1 + BUY_TARGET_PCT)
            stop = px * (1 - BUY_STOP_PCT)
            entry_bar = i
        elif s == SELL:
            position = -1
            entry_price = px
            target = px * (1 - SELL_TARGET_PCT)
            stop = px * (1 + SELL_STOP_PCT)
            entry_bar = i
        if position != 0:
            open_trade = {"entry_i": i, "side": "LONG" if position == 1 else "SHORT",
                          "signal": "BUY" if position == 1 else "SELL",
                          "offsets": offsets_at.get(i, []), "entry_px": px,
                          "target": target, "stop": stop}

        equity[i + 1] = equity[i]

    return equity, trades, trade_log, exit_reasons


def validation_score(events, px_val):
    """t-stat of per-trade net returns on the validation slice (rewards edge AND
    consistency, so one lucky trade can't win). -inf if too few trades."""
    _, signals_at, _ = prepare_signals(events, px_val.index)
    _, trades, _, _ = simulate(px_val.values, signals_at)
    r = np.asarray(trades)
    if len(r) < MIN_VAL_TRADES:
        return -np.inf, len(r)
    sd = max(r.std(ddof=1), 1e-6)
    return float(r.mean() / sd * np.sqrt(len(r))), len(r)


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(datasets, base, bool_cols):
    """
    For each fold (consecutive chunk of the test region):
      1. TUNE  : fit on history before the validation slice, try every
                 (alpha, threshold), score on the validation slice -> pick best.
      2. REFIT : refit with the best alpha on ALL history before the fold.
      3. TEST  : predict the fold's chunk with the best threshold.
    The test chunk is never seen during tuning. A baseline (alpha=1, argmax)
    is walked forward alongside for comparison.
    """
    px = base["Close"]
    n = len(px)
    bar = px.index.to_series().diff().median()          # actual bar length (1m, 5m, ...)
    embargo = FORWARD_BARS * bar                        # labels look this far ahead
    edges = np.linspace(int(n * TRAIN_FRAC), n, N_FOLDS + 1).astype(int)

    def fit_start_for(end_pos):
        if ROLLING_TRAIN_BARS:
            return px.index[max(0, end_pos - ROLLING_TRAIN_BARS)]
        return px.index[0]

    ev_tuned, ev_base, rows = [], [], []
    print(f"\n{'═' * 72}\n  Walk-forward: {N_FOLDS} folds, tuning on last {VAL_FRAC:.0%} of history\n{'═' * 72}")

    for f in range(N_FOLDS):
        t0 = px.index[edges[f]]
        t1 = (px.index[edges[f + 1]] if edges[f + 1] < n
              else px.index[-1] + pd.Timedelta(minutes=1))

        # ── 1. TUNE on a validation slice that ends before the test chunk ─────
        val_pos = int(edges[f] * (1 - VAL_FRAC))
        val_start = px.index[val_pos]
        px_val = px[(px.index >= val_start) & (px.index < t0)]

        best = (-np.inf, BASELINE_ALPHA, None, 0)          # score, alpha, thr, n_trades
        for alpha in ALPHA_GRID:
            models = fit_models(datasets, bool_cols, alpha,
                                fit_start_for(val_pos), val_start - embargo)
            if models is None:
                continue
            scored_val = score_blocks(models, datasets, bool_cols, val_start, t0)
            for thr in THRESH_GRID:
                score, n_tr = validation_score(to_events(scored_val, thr), px_val)
                if score > best[0]:
                    best = (score, alpha, thr, n_tr)

        score, alpha, thr, n_tr = best
        fallback = not np.isfinite(score)
        if fallback:
            print(f"  ⚠ fold {f+1}: no candidate reached {MIN_VAL_TRADES} validation "
                  f"trades - falling back to alpha={BASELINE_ALPHA}, argmax")

        # ── 2. REFIT on all history before this fold, 3. TEST ─────────────────
        fit_end = t0 - embargo
        fit_start = fit_start_for(edges[f])
        models = fit_models(datasets, bool_cols, alpha, fit_start, fit_end)
        models_base = fit_models(datasets, bool_cols, BASELINE_ALPHA, fit_start, fit_end)
        if models is None or models_base is None:
            print(f"  ⚠ fold {f+1}: not enough training data - skipped")
            continue

        ev_tuned.append(to_events(score_blocks(models, datasets, bool_cols, t0, t1), thr))
        ev_base.append(to_events(score_blocks(models_base, datasets, bool_cols, t0, t1), None))

        thr_txt = "argmax" if thr is None else f"P>={thr:.2f}"
        print(f"  Fold {f+1}/{N_FOLDS}  test {t0:%m-%d %H:%M} → {t1:%m-%d %H:%M}  |  "
              f"alpha={alpha:<4}  {thr_txt:<8}  val score={score:6.2f} ({n_tr} trades)"
              + ("  [fallback]" if fallback else ""))
        rows.append({"fold": f + 1, "test_start": t0, "test_end": t1, "alpha": alpha,
                     "threshold": thr, "val_score": score, "val_trades": n_tr,
                     "fallback": fallback})

    def cat(parts):
        parts = [p for p in parts if len(p)]
        return (pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame(columns=["start_ts", "end_ts", "offset", "pred"]))

    cutoff = px.index[edges[0]]
    return cat(ev_tuned), cat(ev_base), cutoff, rows


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def summarise(label, equity, trades, exit_reasons):
    t = np.array(trades)
    wins, losses = t[t > 0], t[t < 0]
    print(f"\n── {label} {'─' * (60 - len(label))}")
    print(f"Trades          : {len(t)}")
    print(f"Win rate        : {(t > 0).mean():.2%}" if len(t) else "Win rate: n/a")
    print(f"Avg win         : {wins.mean():.4%}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss        : {losses.mean():.4%}" if len(losses) else "Avg loss: n/a")
    print(f"Exit breakdown  : {exit_reasons}")
    print(f"Total return    : {equity[-1] - 1:.4%}")
    print(f"Max drawdown    : {(equity / np.maximum.accumulate(equity) - 1).min():.4%}")
    if len(t):
        print(f"Avg trade       : {t.mean():.4%}")


def exit_sanity_check(close):
    """How often each side's own target/stop is reachable within MAX_HOLD_BARS."""
    s = pd.Series(close)
    H = MAX_HOLD_BARS
    up = s[::-1].rolling(H).max()[::-1].shift(-1) / s - 1
    down = 1 - s[::-1].rolling(H).min()[::-1].shift(-1) / s
    up, down = up.dropna(), down.dropna()
    q = [50, 90, 99]
    fmt = lambda a: " / ".join(f"{np.percentile(a, p):.2%}" for p in q)
    print(f"\n  Exit sanity check (any entry, {H}-bar hold):")
    print(f"    max up-move    p50/p90/p99 : {fmt(up)}")
    print(f"    max down-move  p50/p90/p99 : {fmt(down)}")
    print(f"    BUY : target {BUY_TARGET_PCT:.2%} reachable {(up >= BUY_TARGET_PCT).mean():.1%} | "
          f"stop {BUY_STOP_PCT:.2%} reachable {(down >= BUY_STOP_PCT).mean():.1%}")
    print(f"    SELL: target {SELL_TARGET_PCT:.2%} reachable {(down >= SELL_TARGET_PCT).mean():.1%} | "
          f"stop {SELL_STOP_PCT:.2%} reachable {(up >= SELL_STOP_PCT).mean():.1%}")
    for name, a, b in (("BUY", up >= BUY_TARGET_PCT, down >= BUY_STOP_PCT),
                       ("SELL", down >= SELL_TARGET_PCT, up >= SELL_STOP_PCT)):
        if a.mean() < 0.02 or b.mean() < 0.02:
            print(f"    ⚠  {name}: target and/or stop almost never reachable -> mostly timeouts.")


def backtest():
    datasets, base = build_bunched_datasets(return_base=True)
    bool_cols = [c for c in datasets[0].columns if c not in META_COLS]

    ev_tuned, ev_base, cutoff, rows = walk_forward(datasets, base, bool_cols)
    pd.DataFrame(rows).to_csv(PARAMS_CSV, index=False)
    print(f"\nPer-fold settings saved → {PARAMS_CSV}")

    if ENTRY_MODE == "start":
        print("\n  ⚠  ENTRY_MODE='start': entries use a signal that isn't known "
              "until ~4 bars later (lookahead). Results are optimistic.")

    px_test = base["Close"][base.index >= cutoff]
    close = px_test.values
    exit_sanity_check(close)

    runs = {}
    for label, ev in (("BASELINE (alpha=1, argmax)", ev_base), ("TUNED (walk-forward)", ev_tuned)):
        ev_i, signals_at, offsets_at = prepare_signals(ev, px_test.index)
        equity, trades, trade_log, exit_reasons = simulate(close, signals_at, offsets_at)
        summarise(label, equity, trades, exit_reasons)
        runs[label] = (ev_i, equity, trades, trade_log)

    ev_i, equity, trades, trade_log = runs["TUNED (walk-forward)"]
    eq_base = runs["BASELINE (alpha=1, argmax)"][1]

    tl = pd.DataFrame(trade_log)
    if len(tl):
        tl["entry_time"] = px_test.index[tl["entry_i"].values]
        tl["exit_time"] = px_test.index[tl["exit_i"].values]
        tl["bars_held"] = tl["exit_i"] - tl["entry_i"]
        tl = tl[["entry_time", "exit_time", "side", "signal", "offsets", "entry_px",
                 "exit_px", "target", "stop", "reason", "bars_held", "ret",
                 "entry_i", "exit_i"]]
        tl.drop(columns=["entry_i", "exit_i"]).to_csv(TRADES_CSV, index=False)
        print(f"Trade log saved → {TRADES_CSV}")

    taken = set(tl["entry_i"]) if len(tl) else set()
    skipped = ev_i[~ev_i["i"].isin(taken)] if len(ev_i) else ev_i

    plot_trades(px_test, equity, tl, skipped, eq_baseline=eq_base)
    return equity, np.array(trades), tl


EXIT_STYLE = {"target": ("o", "green"), "stop": ("X", "red"), "timeout": ("s", "gray")}


def plot_trades(px_test, equity, tl, skipped, eq_baseline=None):
    """Price chart with every trade drawn on it + equity curve underneath."""
    lo = pd.Timestamp(PLOT_START, tz=px_test.index.tz) if PLOT_START else px_test.index[0]
    hi = pd.Timestamp(PLOT_END, tz=px_test.index.tz) if PLOT_END else px_test.index[-1]

    px = px_test[(px_test.index >= lo) & (px_test.index <= hi)]
    eq = pd.Series(equity[1:], index=px_test.index)
    eq = eq[(eq.index >= lo) & (eq.index <= hi)]
    tr = tl[(tl["exit_time"] >= lo) & (tl["entry_time"] <= hi)] if len(tl) else tl

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(px.index, px.values, color="black", linewidth=0.8, label="Close (1m)")

    if len(tr):
        for r in tr.itertuples():
            ax.plot([r.entry_time, r.exit_time], [r.entry_px, r.exit_px],
                    color="green" if r.ret > 0 else "red", linewidth=2.5, alpha=0.8, zorder=3)
            if DRAW_LEVELS:
                ax.hlines(r.target, r.entry_time, r.exit_time, colors="green",
                          linestyles="dashed", linewidth=0.8)
                ax.hlines(r.stop, r.entry_time, r.exit_time, colors="red",
                          linestyles="dashed", linewidth=0.8)

        longs, shorts = tr[tr["side"] == "LONG"], tr[tr["side"] == "SHORT"]
        ax.scatter(longs["entry_time"], longs["entry_px"], marker="^", s=140,
                   color="limegreen", edgecolor="black", zorder=5, label="Long entry (BUY)")
        ax.scatter(shorts["entry_time"], shorts["entry_px"], marker="v", s=140,
                   color="red", edgecolor="black", zorder=5, label="Short entry (SELL)")

        for reason, (mk, col) in EXIT_STYLE.items():
            sub = tr[tr["reason"] == reason]
            if len(sub):
                ax.scatter(sub["exit_time"], sub["exit_px"], marker=mk, s=90, color=col,
                           edgecolor="black", zorder=5, label=f"Exit: {reason}")

    if SHOW_SKIPPED and len(skipped):
        sk = skipped.copy()
        sk["t"] = px_test.index[sk["i"].values]
        sk = sk[(sk["t"] >= lo) & (sk["t"] <= hi)]
        sk["px"] = px_test.reindex(sk["t"]).values
        for code, mk, lab in ((BUY, "^", "Skipped BUY"), (SELL, "v", "Skipped SELL")):
            s = sk[sk["pred"] == code]
            if len(s):
                ax.scatter(s["t"], s["px"], marker=mk, s=40, facecolors="none",
                           edgecolors="gray", alpha=0.6, zorder=2, label=lab)

    ax.set_xlim(px.index[0], px.index[-1])
    ax.set_title(f"Bunched Naive Bayes (walk-forward tuned) — {len(tr)} trades (entry={ENTRY_MODE})")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    ax2.plot(eq.index, eq.values, color="black", linewidth=1, label="Tuned")
    if eq_baseline is not None:
        b = pd.Series(eq_baseline[1:], index=px_test.index)
        b = b[(b.index >= lo) & (b.index <= hi)]
        ax2.plot(b.index, b.values, color="gray", linestyle="--", linewidth=1,
                 label="Baseline (alpha=1, argmax)")
    ax2.set_title("Equity")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()


if __name__ == "__main__":
    equity, trades, trade_log = backtest()