"""
Walk-Forward Backtest for StockPriceLSTMNetwork (dual-stream)
=============================================================
Usage
-----
    python lstm_backtest.py --model StockPriceLSTMNetwork_<timestamp>.pt

    # Only predict when at least one signal feature is True on the latest bar
    python lstm_backtest.py --model StockPriceLSTMNetwork_<timestamp>.pt --gate
"""

import argparse
import warnings
import datetime

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

from feature_engineering import build_features, FEATURES
from model import StockPriceLSTMNetwork, prepare_price_input, StockPriceLSTMNetworkDualStream

warnings.filterwarnings("ignore")


TICKER          = "AAPL"
WINDOW_SIZE     = 14
PRED_STEPS      = 14
THRESHOLD       = 1.0
GATE_ON_SIGNALS = True   # True → only predict when a signal feature is active
                          # False → always predict

# Signal features = everything except the first column (Close)
BOOL_COLS = [f for f in FEATURES if f != "Close"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare_data() -> pd.DataFrame:
    print(f"[data] Downloading {TICKER} 60d / 5m …")
    df = yf.download(TICKER, period="60d", interval="5m", progress=False, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df.index   = pd.to_datetime(df.index)
    df         = build_features(df)
    df         = df.dropna(subset=FEATURES).copy()
    print(f"[data] {len(df)} bars after cleaning")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_forward(
    model:        StockPriceLSTMNetworkDualStream,
    close_window: np.ndarray,   # (WINDOW_SIZE,)  raw Close prices
    bool_window:  np.ndarray,   # (WINDOW_SIZE,   n_bool) raw 0/1 booleans
    close_scaler: MinMaxScaler,
) -> tuple[np.ndarray, str, float]:
    """
    Roll the dual-stream model forward for PRED_STEPS bars.

    For each step:
      - price stream : prepare_price_input on the current close window
                       (log-returns, z-scored) — shape (W-1, 1)
      - bool stream  : last W-1 rows of the bool window — shape (W-1, n_bool)
      - prediction   : normalised Close for the next bar; inverse-transformed
                       back to price space for the rolling window update

    Boolean signals are held constant at their last observed values for the
    multi-step rollout because we have no way to predict future signal states.
    This is the correct assumption: the model only had past signal state
    available when it was trained.
    """
    model.eval()

    # Normalise close prices into (-1, 1) to match training
    close_norm = close_scaler.transform(close_window.reshape(-1, 1)).flatten()

    preds_norm  = []
    cur_close   = close_norm.copy()         # rolling normalised close window
    last_bools  = bool_window[-1:]          # (1, n_bool) — held constant

    with torch.no_grad():
        for _ in range(PRED_STEPS):
            # Price input: log-returns over the current close window
            price_t = prepare_price_input(
                torch.FloatTensor(cur_close)
            ).unsqueeze(0)                  # (1, W-1, 1)

            # Bool input: repeat last known signal bar for W-1 steps
            bool_seq = np.repeat(last_bools, len(cur_close) - 1, axis=0)  # (W-1, n_bool)
            bool_t   = torch.FloatTensor(bool_seq).unsqueeze(0)            # (1, W-1, n_bool)

            pred_norm = model(price_t, bool_t).item()   # scalar in (-1, 1)
            preds_norm.append(pred_norm)

            # Slide close window forward by one step
            cur_close = np.append(cur_close[1:], pred_norm)

    preds_inv  = close_scaler.inverse_transform(
        np.array(preds_norm).reshape(-1, 1)
    ).flatten()

    pct_change = (preds_inv[-1] - preds_inv[0]) / (abs(preds_inv[0]) + 1e-9) * 100

    if   pct_change >  THRESHOLD: signal = "BUY"
    elif pct_change < -THRESHOLD: signal = "SELL"
    else:                         signal = "HOLD"

    return preds_inv, signal, pct_change


def any_signal_active(bool_window: np.ndarray) -> bool:
    """True if any boolean feature is non-zero on the latest bar."""
    return bool(np.any(bool_window[-1] != 0))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def determine_actual_outcome(actual_closes: np.ndarray) -> str:
    pct = (actual_closes[-1] - actual_closes[0]) / (abs(actual_closes[0]) + 1e-9) * 100
    if pct >  THRESHOLD: return "BUY"
    if pct < -THRESHOLD: return "SELL"
    return "HOLD"


def run_walk_forward_backtest(
    model:           StockPriceLSTMNetworkDualStream,
    df:              pd.DataFrame,
    close_scaler:    MinMaxScaler,
    step:            int  = 1,
    gate_on_signals: bool = False,
) -> pd.DataFrame:
    """
    gate_on_signals=True  — model only predicts when at least one boolean
                            signal feature is True on the latest bar.
                            Skipped windows are recorded as HOLD.
    gate_on_signals=False — model always predicts (original behaviour).
    """
    close_vals = df["Close"].values.astype(float)
    bool_vals  = df[BOOL_COLS].values.astype(float)
    dates      = df.index

    results   = []
    min_start = WINDOW_SIZE + 1
    max_start = len(df) - PRED_STEPS
    total     = len(range(min_start, max_start, step))
    skipped   = 0

    mode_label = "gated (signal required)" if gate_on_signals else "always predict"
    print(f"[backtest] {total} windows  |  step={step}  |  mode={mode_label} …")

    for idx, i in enumerate(range(min_start, max_start, step)):

        close_window  = close_vals[i - WINDOW_SIZE : i]    # (W,)
        bool_window   = bool_vals [i - WINDOW_SIZE : i]    # (W, n_bool)
        actual_window = close_vals[i : i + PRED_STEPS]

        entry_date    = dates[i]
        exit_date     = dates[min(i + PRED_STEPS - 1, len(dates) - 1)]
        entry_price   = actual_window[0]
        exit_price    = actual_window[-1]
        actual_pct    = (exit_price - entry_price) / (abs(entry_price) + 1e-9) * 100
        actual_outcome = determine_actual_outcome(actual_window)

        # ── Gate check ───────────────────────────────────────────────────────
        if gate_on_signals and not any_signal_active(bool_window):
            skipped += 1
            results.append({
                "entry_date":     entry_date,
                "exit_date":      exit_date,
                "entry_price":    round(entry_price, 4),
                "exit_price":     round(exit_price,  4),
                "pred_first":     None,
                "pred_last":      None,
                "pred_pct":       None,
                "actual_pct":     round(actual_pct, 3),
                "signal":         "HOLD",
                "actual_outcome": actual_outcome,
                "correct":        actual_outcome == "HOLD",
                "pnl_pct":        0.0,
                "gated_out":      True,
            })
            continue

        # ── Predict ──────────────────────────────────────────────────────────
        try:
            preds_inv, signal, pred_pct = predict_forward(
                model, close_window, bool_window, close_scaler
            )
        except Exception as exc:
            print(f"  [warn] window {i} skipped: {exc}")
            continue

        # ── P&L ──────────────────────────────────────────────────────────────
        commission = 0.001
        if signal == "BUY":
            pnl_pct = ((exit_price - entry_price) / entry_price - 2 * commission) * 100
        elif signal == "SELL":
            pnl_pct = ((entry_price - exit_price) / entry_price - 2 * commission) * 100
        else:
            pnl_pct = 0.0

        results.append({
            "entry_date":     entry_date,
            "exit_date":      exit_date,
            "entry_price":    round(entry_price, 4),
            "exit_price":     round(exit_price,  4),
            "pred_first":     round(preds_inv[0],  4),
            "pred_last":      round(preds_inv[-1], 4),
            "pred_pct":       round(pred_pct, 3),
            "actual_pct":     round(actual_pct, 3),
            "signal":         signal,
            "actual_outcome": actual_outcome,
            "correct":        signal == actual_outcome,
            "pnl_pct":        round(pnl_pct, 4),
            "gated_out":      False,
        })

        if (idx + 1) % 50 == 0:
            print(f"  … {idx + 1}/{total} done")

    if gate_on_signals:
        print(f"[backtest] {skipped}/{total} windows skipped (no signal active)")
    print(f"[backtest] Complete — {len(results)} windows recorded.\n")
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_print_metrics(results: pd.DataFrame) -> dict:
    total = len(results)
    if total == 0:
        print("[error] No results to report.")
        return {}

    predicted = results[~results["gated_out"]]

    sig_counts = results["signal"].value_counts()
    act_counts = results["actual_outcome"].value_counts()
    overall_acc = predicted["correct"].mean() * 100 if len(predicted) else 0.0

    directional = predicted[predicted["signal"] != "HOLD"]
    dir_acc     = directional["correct"].mean() * 100 if len(directional) else 0.0

    buy_rows  = predicted[predicted["signal"] == "BUY"]
    sell_rows = predicted[predicted["signal"] == "SELL"]
    hold_rows = predicted[predicted["signal"] == "HOLD"]

    buy_acc  = buy_rows["correct"].mean()  * 100 if len(buy_rows)  else float("nan")
    sell_acc = sell_rows["correct"].mean() * 100 if len(sell_rows) else float("nan")
    hold_acc = hold_rows["correct"].mean() * 100 if len(hold_rows) else float("nan")

    traded   = predicted[predicted["signal"] != "HOLD"]
    winners  = traded[traded["pnl_pct"] > 0]
    losers   = traded[traded["pnl_pct"] <= 0]
    win_rate = len(winners) / len(traded) * 100 if len(traded) else 0.0
    avg_win  = winners["pnl_pct"].mean() if len(winners) else 0.0
    avg_loss = losers["pnl_pct"].mean()  if len(losers)  else 0.0
    profit_fac = (
        winners["pnl_pct"].sum() / (abs(losers["pnl_pct"].sum()) + 1e-9)
        if len(losers) else float("inf")
    )
    total_pnl = traded["pnl_pct"].sum()
    avg_pnl   = traded["pnl_pct"].mean() if len(traded) else 0.0

    equity       = 100.0
    equity_curve = [100.0]
    for pnl in traded["pnl_pct"]:
        equity *= (1 + pnl / 100)
        equity_curve.append(equity)
    cum_return  = equity - 100.0
    equity_arr  = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    max_dd      = ((equity_arr - running_max) / running_max * 100).min()

    y_true  = predicted["actual_outcome"].values
    y_pred  = predicted["signal"].values
    present = sorted(set(y_true) | set(y_pred))

    if len(y_true) == 0:
        print("[warn] No windows were predicted — all were gated out.")
        return {}

    gated_count = int(results["gated_out"].sum())

    w = 55
    print("\n" + "═" * w)
    print("  LSTM WALK-FORWARD BACKTEST  —  PERFORMANCE REPORT")
    print("═" * w)
    print(f"\n  Dataset  : {TICKER} · 60d · 5-minute bars")
    print(f"  Windows  : {total}  |  Predicted: {len(predicted)}  |  Gated out: {gated_count}")
    print(f"  Look-back: {WINDOW_SIZE}  |  Horizon: {PRED_STEPS}  |  Threshold: ±{THRESHOLD}%")

    print(f"\n  ── Signal Distribution {'─' * (w - 25)}")
    for lbl in ["BUY", "HOLD", "SELL"]:
        print(f"    {lbl:<6}  predicted: {sig_counts.get(lbl, 0):>5}   actual: {act_counts.get(lbl, 0):>5}")

    print(f"\n  ── Accuracy (predicted windows only) {'─' * (w - 39)}")
    print(f"    Overall accuracy (3-class)    : {overall_acc:>6.1f} %")
    print(f"    Directional accuracy (no HOLD): {dir_acc:>6.1f} %")
    print(f"    BUY  signal accuracy          : {buy_acc:>6.1f} %")
    print(f"    SELL signal accuracy          : {sell_acc:>6.1f} %")
    print(f"    HOLD signal accuracy          : {hold_acc:>6.1f} %")

    print(f"\n  ── P&L (non-HOLD trades, 0.1% commission/side) {'─' * (w - 49)}")
    print(f"    Trades taken      : {len(traded):>6}")
    print(f"    Win rate          : {win_rate:>6.1f} %")
    print(f"    Avg win           : {avg_win:>+6.3f} %")
    print(f"    Avg loss          : {avg_loss:>+6.3f} %")
    print(f"    Profit factor     : {profit_fac:>7.3f}")
    print(f"    Avg P&L per trade : {avg_pnl:>+6.3f} %")
    print(f"    Total P&L (sum)   : {total_pnl:>+6.2f} %")
    print(f"    Cumulative return : {cum_return:>+6.2f} %")
    print(f"    Max drawdown      : {max_dd:>+6.2f} %")

    print(f"\n  ── Confusion Matrix {'─' * (w - 22)}")
    cm     = confusion_matrix(y_true, y_pred, labels=present)
    header = "          " + "  ".join(f"{l:>6}" for l in present)
    print(f"    {header}")
    for i, lbl in enumerate(present):
        print(f"    pred {lbl:<5}  " + "  ".join(f"{v:>6}" for v in cm[i]))

    print(f"\n  ── Classification Report {'─' * (w - 27)}")
    for line in classification_report(y_true, y_pred, labels=present, zero_division=0).split("\n"):
        print(f"    {line}")

    print("═" * w + "\n")

    return dict(
        total_windows    = total,
        gated_out        = gated_count,
        overall_acc      = overall_acc,
        directional_acc  = dir_acc,
        buy_acc          = buy_acc,
        sell_acc         = sell_acc,
        hold_acc         = hold_acc,
        n_trades         = len(traded),
        win_rate         = win_rate,
        avg_win_pct      = avg_win,
        avg_loss_pct     = avg_loss,
        profit_factor    = profit_fac,
        cum_return_pct   = cum_return,
        max_drawdown_pct = max_dd,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True,  help="Path to .pt checkpoint")
    p.add_argument("--step",         type=int, default=5)
    p.add_argument("--gate",         action="store_true", help="Gate predictions on signal activity")
    p.add_argument("--save_results", default="backtest_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    gate = args.gate or GATE_ON_SIGNALS

    df = load_and_prepare_data()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    # The training script saves a dict with architecture params and the scaler.
    # We use those to reconstruct the model exactly as trained.
    print(f"[model] Loading from {args.model} …")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)

    n_bool       = ckpt["n_bool_features"]
    hidden_size  = ckpt["hidden_size"]
    close_scaler = ckpt["close_scaler"]     # fitted MinMaxScaler from training

    model = StockPriceLSTMNetworkDualStream(
        n_bool_features = n_bool,
        hidden_size     = hidden_size,
        output_size     = 1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[model] Loaded. n_bool={n_bool}, hidden={hidden_size}\n")

    results = run_walk_forward_backtest(
        model, df, close_scaler,
        step            = args.step,
        gate_on_signals = gate,
    )
    metrics = compute_and_print_metrics(results)

    results.to_csv(args.save_results, index=False)
    print(f"[output] Results saved → {args.save_results}")
    return results, metrics


if __name__ == "__main__":
    main()