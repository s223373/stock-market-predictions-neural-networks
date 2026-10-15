import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB

from dataset_builder import BUY, HOLD, SELL, FORWARD_BARS
from bunched_datasets import build_bunched_datasets

MAX_HOLD_BARS = 100          # in 1m bars
FEE_BPS = 1.0
TARGET_PCT = 0.1
STOP_PCT = 0.05
TRAIN_FRAC = 0.8

# "start" -> enter at the FIRST 1m bar of the 5-bar sample (what you asked for;
#            note this peeks ~4 min into the future, see notes)
# "end"   -> enter at the LAST 1m bar, i.e. when the sample is actually complete
ENTRY_MODE = "end"

# ── Plot config ──────────────────────────────────────────────────────────────
PLOT_START = None            # e.g. "2026-10-08 09:30"  (None = start of test period)
PLOT_END = None              # e.g. "2026-10-08 16:00"  (None = end of test period)
DRAW_LEVELS = False          # draw each trade's target/stop lines
SHOW_SKIPPED = True          # faint markers for signals that fired but weren't traded
TRADES_CSV = "nb_bunched_trades.csv"
PLOT_PATH = "nb_bunched_trades.png"

META_COLS = ["Close", "target", "start_ts"]


def collect_signals(datasets, base):
    """Train one NB model per offset dataset, return all non-HOLD predictions
    on the shared test period as a single events table."""
    px = base["Close"]
    cutoff = px.index[int(len(px) * TRAIN_FRAC)]
    embargo = pd.Timedelta(minutes=FORWARD_BARS)   # labels look FORWARD_BARS (1m) ahead

    events = []
    for k, ds in datasets.items():
        end_ts = ds.index
        start_ts = pd.DatetimeIndex(ds["start_ts"])

        train = np.asarray(end_ts < (cutoff - embargo))   # purge label overlap
        test = np.asarray(start_ts >= cutoff)

        feats = ds.drop(columns=META_COLS)
        model = BernoulliNB(binarize=None)                # features are already 0/1
        model.fit(feats[train], ds.loc[train, "target"])
        preds = model.predict(feats[test])

        ev = pd.DataFrame({
            "start_ts": start_ts[test],
            "end_ts": end_ts[test],
            "pred": preds,
            "offset": k,
        })
        n_buy, n_sell = (ev["pred"] == BUY).sum(), (ev["pred"] == SELL).sum()
        print(f"  offset {k}: train={train.sum():,}  test={test.sum():,}  "
              f"BUY={n_buy}  SELL={n_sell}")
        events.append(ev[ev["pred"] != HOLD])

    return pd.concat(events, ignore_index=True), cutoff


def backtest():
    datasets, base = build_bunched_datasets(return_base=True)
    events, cutoff = collect_signals(datasets, base)

    if ENTRY_MODE == "start":
        print("\n  ⚠  ENTRY_MODE='start': entries use a signal that isn't known "
              "until ~4 bars later (lookahead). Results are optimistic.")

    px_test = base["Close"][base.index >= cutoff]
    close = px_test.values
    n = len(close)

    # Map each event to the 1m bar where we'd enter
    entry_col = "start_ts" if ENTRY_MODE == "start" else "end_ts"
    events["i"] = px_test.index.get_indexer(events[entry_col])
    events = events[events["i"] >= 0].copy()

    # Any model firing on a bar triggers entry; if models disagree on the
    # same bar (BUY vs SELL), skip it (same spirit as the split -> HOLD rule)
    g = events.groupby("i")["pred"]
    sig = g.first()[g.nunique() == 1]
    signals_at = sig.to_dict()
    offsets_at = events.groupby("i")["offset"].apply(lambda s: sorted(set(s))).to_dict()
    print(f"  {len(signals_at):,} entry opportunities across all {len(datasets)} models\n")

    equity = np.ones(n + 1)
    position = 0
    entry_price = target = stop = None
    entry_bar = 0
    trades = []
    trade_log = []
    open_trade = None
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
            target = px * (1 + TARGET_PCT)
            stop = px * (1 - STOP_PCT)
            entry_bar = i
        elif s == SELL:
            position = -1
            entry_price = px
            target = px * (1 - TARGET_PCT)
            stop = px * (1 + STOP_PCT)
            entry_bar = i
        if position != 0:
            open_trade = {"entry_i": i, "side": "LONG" if position == 1 else "SHORT",
                          "signal": "BUY" if position == 1 else "SELL",
                          "offsets": offsets_at.get(i, []), "entry_px": px,
                          "target": target, "stop": stop}

        equity[i + 1] = equity[i]

    # ── Trade log ────────────────────────────────────────────────────────────
    tl = pd.DataFrame(trade_log)
    if len(tl):
        tl["entry_time"] = px_test.index[tl["entry_i"].values]
        tl["exit_time"] = px_test.index[tl["exit_i"].values]
        tl["bars_held"] = tl["exit_i"] - tl["entry_i"]
        tl = tl[["entry_time", "exit_time", "side", "signal", "offsets", "entry_px",
                 "exit_px", "target", "stop", "reason", "bars_held", "ret",
                 "entry_i", "exit_i"]]
        tl.drop(columns=["entry_i", "exit_i"]).to_csv(TRADES_CSV, index=False)
        print(f"Trade log saved → {TRADES_CSV}\n")

    taken = set(tl["entry_i"]) if len(tl) else set()
    skipped = events[~events["i"].isin(taken)]

    # ── Stats ────────────────────────────────────────────────────────────────
    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades < 0]

    print(f"Trades          : {len(trades)}")
    print(f"Win rate        : {(trades > 0).mean():.2%}" if len(trades) else "Win rate: n/a")
    print(f"Avg win         : {wins.mean():.4%}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss        : {losses.mean():.4%}" if len(losses) else "Avg loss: n/a")
    print(f"Exit breakdown  : {exit_reasons}")
    print(f"Total return    : {equity[-1] - 1:.4%}")
    print(f"Max drawdown    : {(equity / np.maximum.accumulate(equity) - 1).min():.4%}")
    if len(trades):
        print(f"Avg trade       : {trades.mean():.4%}")

    plot_trades(px_test, equity, tl, skipped)
    return equity, trades, tl


EXIT_STYLE = {"target": ("o", "green"), "stop": ("X", "red"), "timeout": ("s", "gray")}


def plot_trades(px_test, equity, tl, skipped):
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
        # entry -> exit segments, green = winner, red = loser
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

    ax.set_xlim(px.index[0], px.index[-1])      # keep trades that straddle the window edge tidy
    ax.set_title(f"Bunched Naive Bayes — {len(tr)} trades (entry={ENTRY_MODE})")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    ax2.plot(eq.index, eq.values, color="black", linewidth=1)
    ax2.set_title("Equity")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()


if __name__ == "__main__":
    equity, trades, trade_log = backtest()