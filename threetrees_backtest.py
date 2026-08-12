import numpy as np
import pandas as pd
import yfinance as yf
from feature_engineering import build_features
from three_trees import build_model, TICKER, TIMEFRAME_PERIODS, PRIMARY_TF, BUY, SELL, HOLD, plot_tree_feature_importances

MAX_HOLD_BARS = 200      # was 50 — give the target more time to actually get hit
# STOP_LOSS_PCT = 0.0015
FEE_BPS = 1.0
TARGET_PULLBACK = 0.1   # shrink target distance by this fraction (toward entry)

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
    bull_edge = pdf[f"breaker_bull_zone_edge_{PRIMARY_TF}"].values   # NEW
    bear_edge = pdf[f"breaker_bear_zone_edge_{PRIMARY_TF}"].values   # NEW

    n = len(combined)
    equity = np.ones(n + 1)
    position = 0
    entry_price = target = stop = None
    entry_bar = 0
    trades = []
    exit_reasons = {"target": 0, "stop": 0, "timeout": 0}

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
                equity[i+1] = equity[i] * (1 + ret - fee)
                exit_reasons["target" if hit_target else "stop" if hit_stop else "timeout"] += 1
                position = 0
                continue
            else:
                equity[i+1] = equity[i]
                continue

        if position == 0:
            if combined[i] == BUY and not np.isnan(bull_tp[i]) and not np.isnan(bull_edge[i]):
                position = 1
                entry_price = px
                target = px + (bull_tp[i] - px) * (1 - TARGET_PULLBACK)
                stop = bull_edge[i]                      # CHANGED — real zone edge, not fixed %
                entry_bar = i
            elif combined[i] == SELL and not np.isnan(bear_tp[i]) and not np.isnan(bear_edge[i]):
                position = -1
                entry_price = px
                target = px - (px - bear_tp[i]) * (1 - TARGET_PULLBACK)
                stop = bear_edge[i]                       # CHANGED
                entry_bar = i
            equity[i+1] = equity[i]
        else:
            equity[i+1] = equity[i]

    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    print("bull_tp non-nan:", np.sum(~np.isnan(bull_tp)))
    print("bull_edge non-nan:", np.sum(~np.isnan(bull_edge)))
    print("bear_tp non-nan:", np.sum(~np.isnan(bear_tp)))
    print("bear_edge non-nan:", np.sum(~np.isnan(bear_edge)))
    print("BUY signals:", np.sum(combined == BUY))
    print("SELL signals:", np.sum(combined == SELL))
    print(f"Trades          : {len(trades)}")
    print(f"Win rate        : {(trades > 0).mean():.2%}" if len(trades) else "n/a")
    print(f"Avg win         : {wins.mean():.4%}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss        : {losses.mean():.4%}" if len(losses) else "Avg loss: n/a")
    print(f"Exit breakdown  : {exit_reasons}")
    print(f"Total return    : {equity[-1] - 1:.4%}")
    print(f"Max drawdown    : {(equity / np.maximum.accumulate(equity) - 1).min():.4%}")
    if len(trades):
        print(f"Avg trade       : {trades.mean():.4%}")
    return equity, trades

if __name__ == "__main__":
    backtest()