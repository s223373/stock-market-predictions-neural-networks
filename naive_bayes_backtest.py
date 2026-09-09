import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset_builder import build_labeled_dataset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

MAX_HOLD_BARS = 100
FEE_BPS = 1.0
TARGET_PCT = 0.05   # +1.25% target — adjust to your instrument's typical move
STOP_PCT = 0.005    # -0.5% stop

def backtest():
    df = build_labeled_dataset()
    X = df.drop(["target", "Close"], axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101, shuffle=False  # keep time order for backtest
    )

    threshold = X_train.median()
    X_train_bin = (X_train > threshold).astype(int)
    X_test_bin = (X_test > threshold).astype(int)

    model = BernoulliNB()
    model.fit(X_train_bin, y_train)
    preds = model.predict(X_test_bin)

    close = df.loc[X_test.index, "Close"].values
    n = len(preds)

    equity = np.ones(n + 1)
    position = 0
    entry_price = target = stop = None
    entry_bar = 0
    trades = []
    trade_exit_reasons = []
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
                trade_exit_reasons.append(reason)
                equity[i+1] = equity[i] * (1 + ret - fee)
                exit_reasons[reason] += 1
                position = 0
                continue
            else:
                equity[i+1] = equity[i]
                continue

        if position == 0 and preds[i] == True:
            position = 1
            entry_price = px
            target = px * (1 + TARGET_PCT)
            stop = px * (1 - STOP_PCT)
            entry_bar = i

        equity[i+1] = equity[i]

    trades = np.array(trades)
    trade_exit_reasons = np.array(trade_exit_reasons)
    wins = trades[trades > 0]
    losses = trades[trades < 0]

    print(f"Trades          : {len(trades)}")
    print(f"Win rate        : {(trades > 0).mean():.2%}" if len(trades) else "n/a")
    print(f"Avg win         : {wins.mean():.4%}" if len(wins) else "Avg win: n/a")
    print(f"Avg loss        : {losses.mean():.4%}" if len(losses) else "Avg loss: n/a")
    print(f"Exit breakdown  : {exit_reasons}")
    print(f"Total return    : {equity[-1] - 1:.4%}")
    print(f"Max drawdown    : {(equity / np.maximum.accumulate(equity) - 1).min():.4%}")
    if len(trades):
        print(f"Avg trade       : {trades.mean():.4%}")

    plot_equity(equity)
    return equity, trades


def plot_equity(equity):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(equity, color="black", linewidth=1)
    ax.set_title("Naive Bayes Strategy Equity Curve")
    plt.tight_layout()
    plt.savefig("nb_equity_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    equity, trades = backtest()