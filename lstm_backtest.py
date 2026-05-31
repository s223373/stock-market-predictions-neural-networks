"""
Walk-Forward Backtest for StockPriceLSTMNetwork
================================================
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
import torch.nn as nn
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from feature_engineering import build_features, FEATURES

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class StockPriceLSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.layer_norm    = nn.LayerNorm(hidden_size)
        self.lstm          = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                     dropout=0.1, batch_first=True)
        self.dropout       = nn.Dropout(p=dropout)
        self.fc1           = nn.Linear(hidden_size, hidden_size // 2)
        self.relu          = nn.ReLU()
        self.fc2           = nn.Linear(hidden_size // 2, output_size)
        self.residual_proj = nn.Linear(input_size, hidden_size)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, seq):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        lstm_out, _  = self.lstm(seq.view(1, len(seq), -1), (h0, c0))
        residual     = self.residual_proj(seq[-1].unsqueeze(0))
        out          = self.layer_norm(lstm_out[:, -1, :] + residual)
        out          = self.relu(self.fc1(out))
        out          = self.dropout(out)
        return self.fc2(out).squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA
# ─────────────────────────────────────────────────────────────────────────────

TICKER           = "AAPL"
WINDOW_SIZE      = 14
PRED_STEPS       = 14
THRESHOLD        = 1.0
HIDDEN_SIZE      = 64
NUM_LAYERS       = 1
DROPOUT          = 0.3
GATE_ON_SIGNALS  = True    # True  → only predict when a signal feature is active
                            # False → always predict

# Signal features = everything except the first column (Close)
SIGNAL_FEATURES = FEATURES[1:]


def load_and_prepare_data() -> pd.DataFrame:
    print(f"[data] Downloading {TICKER} 60d / 5m …")
    df = yf.download(TICKER, period="60d", interval="5m", progress=False, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df.index   = pd.to_datetime(df.index)

    daily = df.groupby(df.index.date).agg(
        day_high=("High", "max"),
        day_low =("Low",  "min"),
    )
    daily.index            = pd.to_datetime(daily.index)
    daily["prev_day_high"] = daily["day_high"].shift(1)
    daily["prev_day_low"]  = daily["day_low"].shift(1)

    df["prev_day_high"]   = pd.to_numeric(df.index.normalize().map(daily["prev_day_high"]), errors="coerce")
    df["prev_day_low"]    = pd.to_numeric(df.index.normalize().map(daily["prev_day_low"]),  errors="coerce")
    df["high_wick_sweep"] = (df["High"] > df["prev_day_high"]) & (df["Close"] < df["prev_day_high"])
    df["low_wick_sweep"]  = (df["Low"]  < df["prev_day_low"])  & (df["Close"] > df["prev_day_low"])

    df = build_features(df)
    df = df.dropna(subset=FEATURES).copy()
    print(f"[data] {len(df)} bars after cleaning")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_forward(model, raw_window, scaler, close_scaler):
    norm_window = scaler.transform(raw_window)
    input_seq   = torch.FloatTensor(norm_window)

    model.eval()
    preds_norm = []
    with torch.no_grad():
        for step in range(PRED_STEPS):
            seq  = input_seq if step == 0 else torch.FloatTensor(last_window)
            pred = model(seq).item()
            preds_norm.append(pred)

            new_row    = seq[-1].clone()
            new_row[0] = pred
            last_window = torch.cat([seq[1:], new_row.unsqueeze(0)], dim=0).numpy()
            input_seq   = torch.FloatTensor(last_window)

    preds_inv  = close_scaler.inverse_transform(
        np.array(preds_norm).reshape(-1, 1)
    ).flatten()

    pct_change = (preds_inv[-1] - preds_inv[0]) / (preds_inv[0] + 1e-9) * 100

    if pct_change > THRESHOLD:   signal = "BUY"
    elif pct_change < -THRESHOLD: signal = "SELL"
    else:                         signal = "HOLD"

    return preds_inv, signal, pct_change


def any_signal_active(raw_window: np.ndarray) -> bool:
    """
    Check whether any signal feature (all columns except index 0 = Close)
    is True / non-zero on the LATEST bar of the input window.
    """
    latest_bar      = raw_window[-1]       # shape (n_features,)
    signal_values   = latest_bar[1:]       # skip Close at index 0
    return bool(np.any(signal_values != 0))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def determine_actual_outcome(actual_closes, threshold=THRESHOLD):
    pct = (actual_closes[-1] - actual_closes[0]) / (actual_closes[0] + 1e-9) * 100
    if pct > threshold:   return "BUY"
    if pct < -threshold:  return "SELL"
    return "HOLD"


def run_walk_forward_backtest(
    model:           StockPriceLSTMNetwork,
    df:              pd.DataFrame,
    step:            int  = 1,
    gate_on_signals: bool = False,   # ← THE SWITCH
) -> pd.DataFrame:
    """
    gate_on_signals=True  — model only predicts when at least one signal
                            feature is True on the latest bar of the window.
                            Windows with no active signal are recorded as
                            HOLD (skipped) without running the network.

    gate_on_signals=False — model always predicts (original behaviour).
    """
    raw    = df[FEATURES].values.astype(float)
    closes = df["Close"].values.astype(float)
    dates  = df.index

    results    = []
    min_train  = WINDOW_SIZE + 1
    max_start  = len(raw) - PRED_STEPS
    total      = len(range(min_train, max_start, step))

    mode_label = "gated (signal required)" if gate_on_signals else "always predict"
    print(f"[backtest] {total} windows  |  step={step}  |  mode={mode_label} …")

    skipped = 0

    for idx, i in enumerate(range(min_train, max_start, step)):

        history      = raw[:i]
        input_window = raw[i - WINDOW_SIZE : i]
        actual_window = closes[i : i + PRED_STEPS]
        entry_date   = dates[i]
        exit_date    = dates[min(i + PRED_STEPS - 1, len(dates) - 1)]
        entry_price  = actual_window[0]
        exit_price   = actual_window[-1]
        actual_pct   = (exit_price - entry_price) / (entry_price + 1e-9) * 100
        actual_outcome = determine_actual_outcome(actual_window)

        # ── Gate check ───────────────────────────────────────────────────────
        if gate_on_signals and not any_signal_active(input_window):
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

        # ── Fit scalers ──────────────────────────────────────────────────────
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(history)
        close_scaler = MinMaxScaler(feature_range=(-1, 1))
        close_scaler.fit(history[:, 0].reshape(-1, 1))

        # ── Predict ──────────────────────────────────────────────────────────
        try:
            preds_inv, signal, pred_pct = predict_forward(
                model, input_window, scaler, close_scaler
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
            "pred_first":     round(preds_inv[0], 4),
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
# 5.  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_print_metrics(results: pd.DataFrame) -> dict:
    total = len(results)
    if total == 0:
        print("[error] No results to report.")
        return {}

    # Exclude windows that were gated out when computing signal accuracy
    predicted = results[~results.get("gated_out", pd.Series(False, index=results.index))]

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

    traded     = predicted[predicted["signal"] != "HOLD"]
    winners    = traded[traded["pnl_pct"] > 0]
    losers     = traded[traded["pnl_pct"] <= 0]
    win_rate   = len(winners) / len(traded) * 100 if len(traded) else 0.0
    avg_win    = winners["pnl_pct"].mean() if len(winners) else 0.0
    avg_loss   = losers["pnl_pct"].mean()  if len(losers)  else 0.0
    profit_fac = (
        winners["pnl_pct"].sum() / (abs(losers["pnl_pct"].sum()) + 1e-9)
        if len(losers) else float("inf")
    )
    total_pnl = traded["pnl_pct"].sum()
    avg_pnl   = traded["pnl_pct"].mean() if len(traded) else 0.0

    equity = 100.0
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
        print("       Try setting GATE_ON_SIGNALS = False to verify the model runs,")
        print("       or check that your signal features are actually firing in the data.")
        return {}


    gated_count = int(results.get("gated_out", pd.Series(False)).sum())

    w = 55
    print("\n" + "═" * w)
    print("  LSTM WALK-FORWARD BACKTEST  —  PERFORMANCE REPORT")
    print("═" * w)
    print(f"\n  Dataset  : {TICKER} · 30-day · 5-minute bars")
    print(f"  Windows  : {total}  |  Predicted: {len(predicted)}  |  Gated out: {gated_count}")
    print(f"  Look-back: {WINDOW_SIZE}  |  Horizon: {PRED_STEPS}  |  Threshold: ±{THRESHOLD}%")
    print(f"  Signal features checked: {', '.join(SIGNAL_FEATURES)}")

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
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        required=True)
    p.add_argument("--step",         type=int, default=5)
    p.add_argument("--save_results", default="backtest_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    df   = load_and_prepare_data()

    print(f"[model] Loading from {args.model} …")
    model = StockPriceLSTMNetwork(len(FEATURES), HIDDEN_SIZE, 1, NUM_LAYERS, DROPOUT)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()
    print("[model] Loaded.\n")

    results = run_walk_forward_backtest(model, df, step=args.step, gate_on_signals=GATE_ON_SIGNALS)
    metrics = compute_and_print_metrics(results)

    results.to_csv(args.save_results, index=False)
    print(f"[output] Results saved → {args.save_results}")
    return results, metrics


if __name__ == "__main__":
    main()