import numpy as np
import pandas as pd
import yfinance as yf
from feature_engineering import build_features
from three_trees import build_model, TICKER, TIMEFRAME_PERIODS, PRIMARY_TF, BUY, SELL, HOLD, plot_tree_feature_importances

MAX_HOLD_BARS = 100
STOP_CAP_PCT = 0.003
FEE_BPS = 1.0
TARGET_PULLBACK = 0.1
MIN_RR = 0.5

EXTENDED_HOLD_BARS = 300   # NEW — how far forward to simulate "held longer" for

def backtest():
    grids, preds, tests, combined = build_model()
    plot_tree_feature_importances(grids)
    primary_test_idx = tests[PRIMARY_TF].index

    raw = yf.download(TICKER, period=TIMEFRAME_PERIODS[PRIMARY_TF],
                       interval=PRIMARY_TF, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    enriched = build_features(raw, period=TIMEFRAME_PERIODS[PRIMARY_TF],
                               interval=PRIMARY_TF, ticker=TICKER)

    pdf = enriched.loc[enriched.index.intersection(primary_test_idx)]
    assert len(pdf) == len(combined), f"{len(pdf)} vs {len(combined)}"
    close = pdf["Close"].values
    high  = pdf["High"].values
    low   = pdf["Low"].values
    bull_tp = pdf[f"breaker_bull_target_price_{PRIMARY_TF}"].values
    bear_tp = pdf[f"breaker_bear_target_price_{PRIMARY_TF}"].values
    bull_edge = pdf[f"breaker_bull_zone_edge_{PRIMARY_TF}"].values
    bear_edge = pdf[f"breaker_bear_zone_edge_{PRIMARY_TF}"].values

    n = len(combined)
    equity = np.ones(n + 1)
    position = 0
    entry_price = target = stop = None
    entry_bar = 0
    trades = []
    trade_exit_reasons = []
    exit_reasons = {"target": 0, "stop": 0, "timeout": 0}
    skipped_bad_rr = 0

    left_on_table = []
    left_on_table_reason = []

    # NEW — per-trade record of what we'd have needed for the "held longer"
    # simulation: entry bar/price/position/stop, filled at entry time.
    trade_records = []

    for i in range(n):
        px = close[i]

        if position != 0:
            hit_target = (position == 1 and high[i] >= target) or (position == -1 and low[i] <= target)
            hit_stop   = (position == 1 and low[i] <= stop) or (position == -1 and high[i] >= stop)
            timed_out  = (i - entry_bar) >= MAX_HOLD_BARS

            if hit_target or hit_stop or timed_out:
                exit_px = target if hit_target else (stop if hit_stop else px)
                ret = (exit_px - entry_price) / entry_price * position
                fee = FEE_BPS / 10000 * 2
                trades.append(ret - fee)
                reason = "target" if hit_target else "stop" if hit_stop else "timeout"
                trade_exit_reasons.append(reason)
                equity[i+1] = equity[i] * (1 + ret - fee)
                exit_reasons[reason] += 1

                window_end = min(entry_bar + MAX_HOLD_BARS, n - 1)
                window_high = high[entry_bar:window_end + 1]
                window_low  = low[entry_bar:window_end + 1]

                if position == 1:
                    best_price = window_high.max()
                    excursion = (best_price - exit_px) / entry_price
                else:
                    best_price = window_low.min()
                    excursion = (exit_px - best_price) / entry_price

                left_on_table.append(excursion)
                left_on_table_reason.append(reason)

                # NEW — record everything needed to replay this trade with
                # a longer hold, stop still active throughout.
                trade_records.append({
                    "entry_bar": entry_bar,
                    "entry_price": entry_price,
                    "position": position,
                    "stop": stop,
                    "target": target,
                    "actual_exit_reason": reason,
                    "actual_ret": ret - fee,
                })

                position = 0
                continue
            else:
                equity[i+1] = equity[i]
                continue

        if position == 0:
            if combined[i] == BUY and not np.isnan(bull_tp[i]) and not np.isnan(bull_edge[i]):
                candidate_target = px + (bull_tp[i] - px) * (1 - TARGET_PULLBACK)
                max_stop_dist = px * STOP_CAP_PCT
                candidate_stop = max(bull_edge[i], px - max_stop_dist)

                reward = candidate_target - px
                risk   = px - candidate_stop

                if reward > 0 and risk > 0 and (reward / risk) >= MIN_RR:
                    position = 1
                    entry_price = px
                    target = candidate_target
                    stop = candidate_stop
                    entry_bar = i
                else:
                    skipped_bad_rr += 1

            elif combined[i] == SELL and not np.isnan(bear_tp[i]) and not np.isnan(bear_edge[i]):
                candidate_target = px - (px - bear_tp[i]) * (1 - TARGET_PULLBACK)
                max_stop_dist = px * STOP_CAP_PCT
                candidate_stop = min(bear_edge[i], px + max_stop_dist)

                reward = px - candidate_target
                risk   = candidate_stop - px

                if reward > 0 and risk > 0 and (reward / risk) >= MIN_RR:
                    position = -1
                    entry_price = px
                    target = candidate_target
                    stop = candidate_stop
                    entry_bar = i
                else:
                    skipped_bad_rr += 1

            equity[i+1] = equity[i]
        else:
            equity[i+1] = equity[i]

    trades = np.array(trades)
    trade_exit_reasons = np.array(trade_exit_reasons)
    left_on_table = np.array(left_on_table)
    left_on_table_reason = np.array(left_on_table_reason)
    wins = trades[trades > 0]
    losses = trades[trades < 0]

    print(f"Trades          : {len(trades)}")
    print(f"Skipped (bad RR): {skipped_bad_rr}")
    print(f"Win rate        : {(trades > 0).mean():.2%}" if len(trades) else "n/a")
    print(f"Avg win         : {wins.mean():.4%}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss        : {losses.mean():.4%}" if len(losses) else "Avg loss: n/a")
    print(f"Exit breakdown  : {exit_reasons}")

    for reason in ["target", "stop", "timeout"]:
        mask = (trade_exit_reasons == reason) & (trades < 0)
        if mask.sum():
            print(f"  Avg loss ({reason:8s}): {trades[mask].mean():.4%}  (n={mask.sum()})")

    print(f"Total return    : {equity[-1] - 1:.4%}")
    print(f"Max drawdown    : {(equity / np.maximum.accumulate(equity) - 1).min():.4%}")
    if len(trades):
        print(f"Avg trade       : {trades.mean():.4%}")

    print(f"\n{'─' * 50}")
    print(f"  Left-on-the-table analysis (raw excursion, ignores stop)")
    print(f"{'─' * 50}")
    for reason in ["target", "stop", "timeout"]:
        mask = left_on_table_reason == reason
        if mask.sum():
            vals = left_on_table[mask]
            print(f"  {reason:8s}  n={mask.sum():4d}  "
                  f"avg={vals.mean():.4%}  median={np.median(vals):.4%}  "
                  f"max={vals.max():.4%}")

    # ── NEW: "held longer" replay — stop stays live the whole time ────────
    hl_extra_ret = []          # extra return vs. actual, if held to EXTENDED_HOLD_BARS
    hl_would_be_stopped = []   # would the extended hold have gotten stopped out instead?
    hl_reason = []             # original exit reason, for grouping

    for rec in trade_records:
        eb   = rec["entry_bar"]
        pos  = rec["position"]
        stp  = rec["stop"]
        ep   = rec["entry_price"]

        window_end = min(eb + EXTENDED_HOLD_BARS, n - 1)
        # walk bar-by-bar from entry so the stop is checked in order —
        # this is what makes it different from the raw excursion above
        stopped_out = False
        best_reach = ep  # worst case, no movement
        for j in range(eb, window_end + 1):
            if pos == 1:
                if low[j] <= stp:
                    stopped_out = True
                    exit_px = stp
                    break
                best_reach = max(best_reach, high[j])
            else:
                if high[j] >= stp:
                    stopped_out = True
                    exit_px = stp
                    break
                best_reach = min(best_reach, low[j]) if best_reach == ep else min(best_reach, low[j])

        if stopped_out:
            extended_ret = (exit_px - ep) / ep * pos - (FEE_BPS / 10000 * 2)
        else:
            # never stopped within the extended window — exit at best price reached
            extended_ret = (best_reach - ep) / ep * pos - (FEE_BPS / 10000 * 2)

        hl_extra_ret.append(extended_ret - rec["actual_ret"])
        hl_would_be_stopped.append(stopped_out)
        hl_reason.append(rec["actual_exit_reason"])

    hl_extra_ret = np.array(hl_extra_ret)
    hl_would_be_stopped = np.array(hl_would_be_stopped)
    hl_reason = np.array(hl_reason)

    print(f"\n{'─' * 50}")
    print(f"  Held-longer replay (stop stays live, EXTENDED_HOLD_BARS={EXTENDED_HOLD_BARS})")
    print(f"{'─' * 50}")
    for reason in ["target", "stop", "timeout"]:
        mask = hl_reason == reason
        if mask.sum():
            vals = hl_extra_ret[mask]
            stopped_pct = hl_would_be_stopped[mask].mean()
            print(f"  {reason:8s}  n={mask.sum():4d}  "
                  f"avg extra ret={vals.mean():.4%}  median={np.median(vals):.4%}  "
                  f"would-still-get-stopped={stopped_pct:.1%}")

    target_mask = hl_reason == "target"
    if target_mask.sum():
        avg_extra = hl_extra_ret[target_mask].mean()
        pct_positive = (hl_extra_ret[target_mask] > 0).mean()
        print(f"\n  On TARGET-hit trades: holding longer would have added an average\n"
              f"  of {avg_extra:.4%} return, and {pct_positive:.1%} of the time holding\n"
              f"  longer beat the actual outcome — even accounting for the stop\n"
              f"  staying live the whole way.")
        print(f"  If avg extra return is meaningfully positive, your target is too\n"
              f"  conservative — consider raising TARGET_PULLBACK toward 0.")

    return equity, trades, left_on_table, left_on_table_reason, hl_extra_ret, hl_reason

if __name__ == "__main__":
    backtest()