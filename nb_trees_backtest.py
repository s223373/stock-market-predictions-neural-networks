"""
dual_model_backtest.py
======================
Trades only when the bunched Naive Bayes (nb_bunched_backtest.py) and the
three-trees model (three_trees.py) AGREE, allowing a lookback window:

    trees said BUY 3 candles ago  +  NB says BUY now   ->  BUY (at NB's candle)

Rules
-----
  * Both models are mapped onto the three-trees candle timeline (PRIMARY_TF).
    NB events (1m bunches) are assigned to the candle that contains their
    END bar, so nothing is used before it is actually known.
  * Signals agree when both are the same side and no more than LOOKBACK_CANDLES
    apart. The trade fires on the LATER signal's candle (close of that candle).
  * Each signal is used once (consumed) so a stale tree signal can't keep
    re-confirming new NB signals.
  * A newer opposite signal from the same model replaces its older one.
  * TRIGGER: "either" (whoever fires second), "nb" (NB must be the trigger),
    or "trees" (trees must be the trigger).
  * LOSS COOLDOWN: after a losing trade closes, signals within the next
    LOSS_COOLDOWN_BARS candles are ignored (the model waits). Signals are not
    queued - after the cooldown only a fresh agreement can open a trade.
    COOLDOWN_PER_SIDE = True  -> only the side that lost is paused
    COOLDOWN_PER_SIDE = False -> a loss pauses both sides
  * BAR TUNING: tune_bars() grid-searches lookback / max-hold / cooldown on the
    first TUNE_TRAIN_FRAC of the overlap and re-tests each pick on the rest.

Exits
-----
  EXIT_MODE = "fixed_pct" : side-specific BUY_/SELL_ TARGET and STOP pct around entry
  EXIT_MODE = "breaker"   : the three-trees breaker target / zone-edge stop
                            logic from your three_trees backtest

Target/stop are checked on High/Low; if both hit in one candle the stop wins.

Outputs: dual_model_trades.csv, dual_model_backtest.png
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from feature_engineering import build_features
from three_trees import build_model, TICKER as TREE_TICKER, TIMEFRAME_PERIODS, PRIMARY_TF, BUY, SELL, HOLD
from bunched_datasets import build_bunched_datasets
from dataset_builder import TICKER as NB_TICKER
import nb_bunched_backtest as nb

# ── Agreement config ─────────────────────────────────────────────────────────
LOOKBACK_CANDLES = 5          # max gap (in PRIMARY_TF candles) between the two signals
TRIGGER = "either"            # "either" | "nb" | "trees"
NB_USE_TUNED = True           # True = walk-forward tuned NB, False = baseline (alpha=1, argmax)
LOOKBACK_SWEEP = [0, 1, 2, 3, 5, 8, 12]   # printed comparison table

# ── Trade config ─────────────────────────────────────────────────────────────
EXIT_MODE = "fixed_pct"       # "fixed_pct" | "breaker"
# Side-specific exits from signal_explorer (best TRAIN pair per side).
# NOTE: tuned on 1m bars - re-run signal_explorer on PRIMARY_TF if it isn't 1m.
BUY_TARGET_PCT = 0.0040       # +0.40%
BUY_STOP_PCT = 0.0015         # -0.15%
SELL_TARGET_PCT = 0.0055      # -0.55%
SELL_STOP_PCT = 0.0035        # +0.35%
MAX_HOLD_BARS = 50            # in PRIMARY_TF candles
FEE_BPS = 1.0                 # per side
LOSS_COOLDOWN_BARS = 50       # after a losing trade, skip signals for this many candles (0 = off)
COOLDOWN_PER_SIDE = True      # True = pause only the side that lost | False = pause both sides

# ── Bar tuning (lookback / hold / cooldown) ──────────────────────────────────
RUN_TUNING = True
AUTO_APPLY_TUNED = False      # True = use the best train pick for the main run below
TUNE_TRAIN_FRAC = 0.60
TUNE_MIN_TRADES = 15          # picks with fewer TRAIN trades are ignored
LOOKBACK_GRID = [0, 3, 5, 8]
HOLD_GRID = [25, 50, 75, 100]
COOLDOWN_GRID = [0, 10, 25, 50, 100]

# breaker-mode settings (same meaning as in your three_trees backtest)
STOP_CAP_PCT = 0.005
TARGET_PULLBACK = 0.1
MIN_RR = 1.0

TRADES_CSV = "dual_model_trades.csv"
PLOT_PATH = "dual_model_backtest.png"


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _pandas_freq(tf: str) -> str:
    return tf[:-1] + {"m": "min", "h": "h", "d": "D"}[tf[-1]]


def _match_tz(idx: pd.DatetimeIndex, ref: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if ref.tz is None:
        return idx.tz_convert(None) if idx.tz is not None else idx
    return idx.tz_localize(ref.tz) if idx.tz is None else idx.tz_convert(ref.tz)


def load_tree_side():
    """Three-trees signals + OHLC/feature frame on the primary timeframe."""
    grids, preds, tests, combined = build_model()
    test_idx = tests[PRIMARY_TF].index

    raw = yf.download(TREE_TICKER, period=TIMEFRAME_PERIODS[PRIMARY_TF],
                      interval=PRIMARY_TF, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    enriched = build_features(raw, period=TIMEFRAME_PERIODS[PRIMARY_TF],
                              interval=PRIMARY_TF, ticker=TREE_TICKER)
    pdf = enriched.loc[enriched.index.intersection(test_idx)]
    assert len(pdf) == len(combined), f"{len(pdf)} vs {len(combined)}"
    tree_sig = pd.Series(np.asarray(combined), index=pdf.index)
    return pdf, tree_sig


def load_nb_side(candle_index: pd.DatetimeIndex):
    """NB events mapped onto candle positions. Returns (signal array, test-start ts)."""
    if nb.ENTRY_MODE != "end":
        print("  ⚠ nb.ENTRY_MODE != 'end' – dual model always uses each bunch's END bar.")

    datasets, base = build_bunched_datasets(return_base=True)
    bool_cols = [c for c in datasets[0].columns if c not in nb.META_COLS]
    ev_tuned, ev_base, cutoff, _ = nb.walk_forward(datasets, base, bool_cols)
    ev = (ev_tuned if NB_USE_TUNED else ev_base).copy()

    sig = np.full(len(candle_index), HOLD, dtype=int)
    if len(ev) == 0:
        return sig, cutoff

    end_ts = pd.DatetimeIndex(ev["end_ts"])
    end_ts = _match_tz(end_ts, candle_index)
    ev["candle"] = end_ts.floor(_pandas_freq(PRIMARY_TF))
    ev["i"] = candle_index.get_indexer(ev["candle"])
    ev = ev[ev["i"] >= 0]

    g = ev.groupby("i")["pred"]
    agreed = g.first()[g.nunique() == 1]          # same-candle BUY/SELL clash -> dropped
    sig[agreed.index.values] = agreed.values.astype(int)
    return sig, cutoff


# ─────────────────────────────────────────────────────────────────────────────
# AGREEMENT LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def agreement_signals(nb_sig, tree_sig, lookback, trigger):
    """Return {candle_i: BUY/SELL} where both models agree within `lookback` candles."""
    fires = {}
    last_nb = last_tr = None            # [index, side, used]
    for i in range(len(nb_sig)):
        nb_new = nb_sig[i] != HOLD
        tr_new = tree_sig[i] != HOLD
        if nb_new:
            last_nb = [i, nb_sig[i], False]
        if tr_new:
            last_tr = [i, tree_sig[i], False]
        if not (nb_new or tr_new) or last_nb is None or last_tr is None:
            continue
        if trigger == "nb" and not nb_new:
            continue
        if trigger == "trees" and not tr_new:
            continue
        if last_nb[2] or last_tr[2]:
            continue
        if last_nb[1] == last_tr[1] and abs(last_nb[0] - last_tr[0]) <= lookback:
            fires[i] = int(last_nb[1])
            last_nb[2] = last_tr[2] = True
    return fires


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def entry_levels(pdf_cols, i, side, px):
    """(target, stop) for an entry at candle i, or None to skip the trade."""
    if EXIT_MODE == "fixed_pct":
        if side == BUY:
            return px * (1 + BUY_TARGET_PCT), px * (1 - BUY_STOP_PCT)
        return px * (1 - SELL_TARGET_PCT), px * (1 + SELL_STOP_PCT)

    bull_tp, bear_tp, bull_edge, bear_edge = pdf_cols
    if side == BUY:
        if np.isnan(bull_tp[i]) or np.isnan(bull_edge[i]):
            return None
        tgt = px + (bull_tp[i] - px) * (1 - TARGET_PULLBACK)
        stp = max(bull_edge[i], px - px * STOP_CAP_PCT)
        reward, risk = tgt - px, px - stp
    else:
        if np.isnan(bear_tp[i]) or np.isnan(bear_edge[i]):
            return None
        tgt = px - (px - bear_tp[i]) * (1 - TARGET_PULLBACK)
        stp = min(bear_edge[i], px + px * STOP_CAP_PCT)
        reward, risk = px - tgt, stp - px
    if reward <= 0 or risk <= 0 or reward / risk < MIN_RR:
        return None
    return tgt, stp


def simulate(pdf, signals_at, cooldown=None, hold=None):
    cooldown = LOSS_COOLDOWN_BARS if cooldown is None else cooldown
    hold = MAX_HOLD_BARS if hold is None else hold
    close, high, low = pdf["Close"].values, pdf["High"].values, pdf["Low"].values
    pdf_cols = None
    if EXIT_MODE == "breaker":
        pdf_cols = (pdf[f"breaker_bull_target_price_{PRIMARY_TF}"].values,
                    pdf[f"breaker_bear_target_price_{PRIMARY_TF}"].values,
                    pdf[f"breaker_bull_zone_edge_{PRIMARY_TF}"].values,
                    pdf[f"breaker_bear_zone_edge_{PRIMARY_TF}"].values)

    n = len(close)
    equity = np.ones(n + 1)
    pos, entry_bar, entry_px, target, stop = 0, 0, 0.0, 0.0, 0.0
    log, skipped_rr, skipped_cd = [], 0, 0
    last_loss = {1: -10**9, -1: -10**9}      # per side (1=long, -1=short)
    last_loss_any = -10**9
    fee = FEE_BPS / 10000 * 2

    for i in range(n):
        equity[i + 1] = equity[i]
        if pos != 0:
            hit_s = (pos == 1 and low[i] <= stop) or (pos == -1 and high[i] >= stop)
            hit_t = (pos == 1 and high[i] >= target) or (pos == -1 and low[i] <= target)
            timeout = (i - entry_bar) >= hold
            if hit_s or hit_t or timeout:
                if hit_s:   reason, exit_px = "stop", stop       # stop wins ties
                elif hit_t: reason, exit_px = "target", target
                else:       reason, exit_px = "timeout", close[i]
                ret = (exit_px - entry_px) / entry_px * pos - fee
                equity[i + 1] = equity[i] * (1 + ret)
                log.append({"entry_i": entry_bar, "exit_i": i, "side": "LONG" if pos == 1 else "SHORT",
                            "entry_px": entry_px, "exit_px": exit_px, "target": target,
                            "stop": stop, "reason": reason, "bars_held": i - entry_bar, "ret": ret})
                if ret < 0:
                    last_loss[pos] = i
                    last_loss_any = i
                pos = 0
            continue

        side = signals_at.get(i)
        if side in (BUY, SELL):
            ref = last_loss[1 if side == BUY else -1] if COOLDOWN_PER_SIDE else last_loss_any
            if cooldown and (i - ref) <= cooldown:
                skipped_cd += 1
                continue
            lv = entry_levels(pdf_cols, i, side, close[i])
            if lv is None:
                skipped_rr += 1
                continue
            pos = 1 if side == BUY else -1
            entry_bar, entry_px = i, close[i]
            target, stop = lv

    return equity, pd.DataFrame(log), {"levels": skipped_rr, "cooldown": skipped_cd}


def summarise(label, equity, tl, skipped):
    print(f"\n── {label} {'─' * (58 - len(label))}")
    if tl.empty:
        print(f"  No trades.   (skipped: {skipped})")
        return
    r = tl["ret"].values
    print(f"Trades        : {len(tl)}   (skipped -> bad levels: {skipped['levels']}, "
          f"loss cooldown: {skipped['cooldown']})")
    print(f"Win rate      : {(r > 0).mean():.2%}")
    print(f"Avg trade     : {r.mean():.4%}   avg win {r[r > 0].mean() if (r > 0).any() else 0:.4%}"
          f"   avg loss {r[r < 0].mean() if (r < 0).any() else 0:.4%}")
    print(f"Exits         : {tl['reason'].value_counts().to_dict()}   avg hold {tl['bars_held'].mean():.1f} candles")
    print(f"Total return  : {equity[-1] - 1:.4%}")
    print(f"Max drawdown  : {(equity / np.maximum.accumulate(equity) - 1).min():.4%}")
    side_report(tl)


def side_report(tl):
    """Win rate and average win / loss, split BUY (long) vs SELL (short). Returns are net of fees."""
    print(f"\n  {'side':<10}{'trades':>7}{'win%':>8}{'avg win':>10}{'avg loss':>10}"
          f"{'payoff':>8}{'avg trade':>11}{'total':>9}   exits (T/S/timeout)")
    for name, side in (("BUY/long", "LONG"), ("SELL/short", "SHORT")):
        t = tl[tl["side"] == side]
        if t.empty:
            print(f"  {name:<10}{0:>7}")
            continue
        r = t["ret"].values
        w, l = r[r > 0], r[r < 0]
        aw = w.mean() if len(w) else 0.0
        al = l.mean() if len(l) else 0.0
        payoff = f"{abs(aw / al):.2f}" if al else "n/a"
        ex = t["reason"].value_counts()
        print(f"  {name:<10}{len(t):>7}{(r > 0).mean():>8.1%}{aw:>10.3%}{al:>10.3%}"
              f"{payoff:>8}{r.mean():>11.4%}{r.sum():>9.2%}   "
              f"{ex.get('target', 0)}/{ex.get('stop', 0)}/{ex.get('timeout', 0)}")


def _split_signals(sigs, cut):
    tr = {i: v for i, v in sigs.items() if i < cut}
    te = {i - cut: v for i, v in sigs.items() if i >= cut}
    return tr, te


def tune_bars(pdf, nb_sig, tree_sig):
    """
    Grid-search lookback / max-hold / loss-cooldown (all in candles).
    Ranked by t-stat of per-trade net return on the TRAIN slice (edge x consistency);
    each pick is then re-run on the untouched TEST slice. Returns best (L, H, C) or None.
    """
    cut = int(len(pdf) * TUNE_TRAIN_FRAC)
    train, test = pdf.iloc[:cut], pdf.iloc[cut:]
    rows = []
    for L in LOOKBACK_GRID:
        sigs = agreement_signals(nb_sig, tree_sig, L, TRIGGER)
        s_tr, s_te = _split_signals(sigs, cut)
        for H in HOLD_GRID:
            for C in COOLDOWN_GRID:
                eq, tl, _ = simulate(train, s_tr, cooldown=C, hold=H)
                if len(tl) < TUNE_MIN_TRADES:
                    continue
                r = tl["ret"].values
                t_stat = r.mean() / max(r.std(ddof=1), 1e-9) * np.sqrt(len(r))
                eq2, tl2, _ = simulate(test, s_te, cooldown=C, hold=H)
                rows.append({
                    "lookback": L, "hold": H, "cooldown": C,
                    "tr_trades": len(tl), "tr_win%": 100 * (r > 0).mean(),
                    "tr_avg%": 100 * r.mean(), "tr_tstat": t_stat,
                    "te_trades": len(tl2),
                    "te_win%": 100 * (tl2.ret > 0).mean() if len(tl2) else np.nan,
                    "te_avg%": 100 * tl2.ret.mean() if len(tl2) else np.nan,
                })
    print(f"\n  Bar tuning  (train = first {TUNE_TRAIN_FRAC:.0%} of overlap, "
          f"per-side cooldown={COOLDOWN_PER_SIDE})")
    if not rows:
        print(f"  No combination reached {TUNE_MIN_TRADES} train trades - widen the grids or add data.")
        return None
    df = pd.DataFrame(rows).sort_values("tr_tstat", ascending=False)
    print(df.head(8).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("  (te_* columns are out-of-sample; if they collapse vs tr_*, the pick is noise.)")
    b = df.iloc[0]
    return int(b["lookback"]), int(b["hold"]), int(b["cooldown"])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global LOOKBACK_CANDLES, MAX_HOLD_BARS, LOSS_COOLDOWN_BARS
    if NB_TICKER != TREE_TICKER:
        print(f"  ⚠ Different tickers: NB={NB_TICKER}, trees={TREE_TICKER}. Signals won't line up.")

    pdf, tree_s = load_tree_side()
    nb_sig, nb_cutoff = load_nb_side(pdf.index)
    tree_sig = tree_s.values.astype(int)

    # Only the region where BOTH models produce out-of-sample signals is meaningful
    nb_cutoff = _match_tz(pd.DatetimeIndex([nb_cutoff]), pdf.index)[0]
    keep = pdf.index >= nb_cutoff
    print(f"\n  Trees test range : {pdf.index[0]} → {pdf.index[-1]}  ({len(pdf):,} candles)")
    print(f"  NB test starts   : {nb_cutoff}")
    print(f"  Overlap          : {keep.sum():,} candles")
    if keep.sum() < 50:
        print("  ⚠ Overlap is tiny – the two models' test regions barely coincide. "
              "Align their train/test splits (or extend the data) for meaningful results.")
        return
    pdf, nb_sig, tree_sig = pdf[keep], nb_sig[keep], tree_sig[keep]

    print(f"  NB signals       : {(nb_sig != HOLD).sum():,}")
    print(f"  Trees signals    : {(tree_sig != HOLD).sum():,}")

    if RUN_TUNING:
        best = tune_bars(pdf, nb_sig, tree_sig)
        if best and AUTO_APPLY_TUNED:
            LOOKBACK_CANDLES, MAX_HOLD_BARS, LOSS_COOLDOWN_BARS = best
            print(f"  → Applying tuned: lookback={best[0]}, hold={best[1]}, cooldown={best[2]}")

    # lookback sweep (cheap, shows how sensitive agreement is to the window)
    print(f"\n  Lookback sweep (trigger={TRIGGER}, exits={EXIT_MODE})")
    print(f"  {'L':>3} {'agree':>6} {'trades':>7} {'win%':>7} {'avg trade':>10} {'total':>9}")
    for L in LOOKBACK_SWEEP:
        f = agreement_signals(nb_sig, tree_sig, L, TRIGGER)
        eq, tl, _ = simulate(pdf, f)
        if tl.empty:
            print(f"  {L:>3} {len(f):>6} {0:>7}")
        else:
            print(f"  {L:>3} {len(f):>6} {len(tl):>7} {(tl.ret > 0).mean():>7.1%} "
                  f"{tl.ret.mean():>10.4%} {eq[-1] - 1:>9.3%}")

    # main run + single-model references on identical timeline / exits
    dual = agreement_signals(nb_sig, tree_sig, LOOKBACK_CANDLES, TRIGGER)
    nb_only = {i: int(s) for i, s in enumerate(nb_sig) if s != HOLD}
    tr_only = {i: int(s) for i, s in enumerate(tree_sig) if s != HOLD}

    runs = {}
    for label, sigs, cd in (("DUAL (agreement)", dual, None),
                            ("DUAL no cooldown", dual, 0),
                            ("NB only", nb_only, None),
                            ("Trees only", tr_only, None)):
        eq, tl, sk = simulate(pdf, sigs, cooldown=cd)
        summarise(label, eq, tl, sk)
        runs[label] = (eq, tl)

    eq, tl = runs["DUAL (agreement)"]
    if len(tl):
        out = tl.copy()
        out["entry_time"] = pdf.index[out["entry_i"].values]
        out["exit_time"] = pdf.index[out["exit_i"].values]
        out.drop(columns=["entry_i", "exit_i"]).to_csv(TRADES_CSV, index=False)
        print(f"\nTrade log saved → {TRADES_CSV}")

    plot(pdf, nb_sig, tree_sig, tl, runs)


def plot(pdf, nb_sig, tree_sig, tl, runs):
    idx, close = pdf.index, pdf["Close"].values
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(idx, close, color="black", linewidth=0.8, label="Close")

    # faint individual model signals
    for sig, mk_off, name in ((nb_sig, 0, "NB"), (tree_sig, 0, "Trees")):
        for code, mk, col in ((BUY, "^", "green"), (SELL, "v", "red")):
            m = sig == code
            if m.any():
                ax.scatter(idx[m], close[m], marker=mk, s=25, facecolors="none",
                           edgecolors=col, alpha=0.35, label=f"{name} {'BUY' if code == BUY else 'SELL'}")

    if len(tl):
        for r in tl.itertuples():
            ax.plot([idx[r.entry_i], idx[r.exit_i]], [r.entry_px, r.exit_px],
                    color="green" if r.ret > 0 else "red", linewidth=2.5, zorder=3)
        L, S = tl[tl.side == "LONG"], tl[tl.side == "SHORT"]
        ax.scatter(idx[L.entry_i.values], L.entry_px, marker="^", s=150, color="limegreen",
                   edgecolor="black", zorder=5, label="DUAL long")
        ax.scatter(idx[S.entry_i.values], S.entry_px, marker="v", s=150, color="red",
                   edgecolor="black", zorder=5, label="DUAL short")

    ax.set_title(f"Dual model: NB + three trees agree within {LOOKBACK_CANDLES} candles "
                 f"(trigger={TRIGGER}, exits={EXIT_MODE})")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.2)

    for label, (eq, _) in runs.items():
        ax2.plot(idx, eq[1:], label=label, linewidth=1.4 if label.startswith("DUAL") else 0.9)
    ax2.set_title("Equity")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()