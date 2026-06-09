"""
Walk-Forward Backtest for StockPriceLSTMNetwork (dual-stream)
=============================================================
Usage
-----
    python lstm_backtest.py --model StockPriceLSTMNetwork_<timestamp>.pt

    # Only predict when at least one signal feature is True on the latest bar
    python lstm_backtest.py --model StockPriceLSTMNetwork_<timestamp>.pt --gate

    # Include news sentiment features (requires FINNHUB_API_KEY env var)
    python lstm_backtest.py --model StockPriceLSTMNetwork_<timestamp>.pt --sentiment
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
from model import StockPriceLSTMNetwork, prepare_price_input

warnings.filterwarnings("ignore")


TICKER          = "AAPL"
MODEL_PATH      = "StockPriceLSTMNetwork_2026-06-08_22-11-48.pt"  # ← set this to your .pt file
WINDOW_SIZE     = 14
PRED_STEPS      = 14
THRESHOLD       = 1.0
GATE_ON_SIGNALS = True
USE_SENTIMENT   = False   # set True if you have a FINNHUB_API_KEY and want news features
PERIOD          = "60d"
INTERVAL        = "5m"
STEP            = 5       # bars to advance between windows (smaller = more windows, slower)
SAVE_RESULTS    = "backtest_results.csv"

def _get_bool_cols(df: pd.DataFrame) -> list:
    """
    Return the boolean feature columns that are both in FEATURES and
    actually present in df. Sentiment columns are excluded when
    USE_SENTIMENT=False since they were never added to the dataframe.
    """
    return [f for f in FEATURES if f != "Close" and f in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA + SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare_data(use_sentiment: bool = False) -> pd.DataFrame:
    print(f"[data] Downloading {TICKER} {PERIOD} / {INTERVAL} …")
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL,
                     progress=False, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df.index   = pd.to_datetime(df.index)
    df         = build_features(df, period=PERIOD, interval=INTERVAL)
    existing   = [c for c in FEATURES if c in df.columns]
    df         = df.dropna(subset=existing).copy()

    # ── Sentiment ─────────────────────────────────────────────────────────────
    # Pre-compute the full sentiment series ONCE here, before the backtest loop.
    # This is the only correct approach:
    #   - Computing inside the loop would re-fetch on every window (slow + wrong)
    #   - The series is aligned bar-by-bar using strict < timestamps,
    #     so there is zero lookahead even though we compute it upfront
    if use_sentiment:
        sentiment_cols = [
            "news_bull", "news_bear",
            "news_sentiment_strong_bull", "news_sentiment_strong_bear",
        ]
        # Check whether sentiment was already added by build_features
        if not all(c in df.columns for c in sentiment_cols):
            try:
                from news_sentiment import SentimentPipeline
                pipe = SentimentPipeline(TICKER, days_back=70)  # buffer beyond 60d
                df   = pipe.add_sentiment_features(df)
                print(f"[sentiment] Added to backtest dataframe.")
            except Exception as e:
                print(f"[sentiment] Could not load — filling with 0. Reason: {e}")
                for col in sentiment_cols:
                    if col not in df.columns:
                        df[col] = 0.0
        else:
            print("[sentiment] Sentiment columns already present in dataframe.")

    print(f"[data] {len(df)} bars after cleaning")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_forward(
    model:        StockPriceLSTMNetwork,
    close_window: np.ndarray,   # (WINDOW_SIZE,)  normalised Close prices
    bool_window:  np.ndarray,   # (WINDOW_SIZE,   n_bool) raw 0/1 booleans
    close_scaler: MinMaxScaler,
) -> tuple[np.ndarray, str, float]:
    """
    Roll the dual-stream model forward for PRED_STEPS bars.

    Sentiment handling in the rollout
    ----------------------------------
    Sentiment columns are part of bool_window and are held constant at their
    last observed value for the multi-step rollout. This is correct: we have
    no way to predict future news, so the most recent known sentiment state
    is the best forward estimate. The model learned this pattern during
    training — sentiment persisting for several bars and then decaying.
    """
    model.eval()
    close_norm = close_scaler.transform(close_window.reshape(-1, 1)).flatten()

    preds_norm = []
    cur_close  = close_norm.copy()
    last_bools = bool_window[-1:]          # (1, n_bool) — held constant

    with torch.no_grad():
        for _ in range(PRED_STEPS):
            price_t = prepare_price_input(
                torch.FloatTensor(cur_close)
            ).unsqueeze(0)                 # (1, W-1, 1)

            bool_seq = np.repeat(last_bools, len(cur_close) - 1, axis=0)
            bool_t   = torch.FloatTensor(bool_seq).unsqueeze(0)

            pred_norm = model(price_t, bool_t).item()
            preds_norm.append(pred_norm)
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
    model:           StockPriceLSTMNetwork,
    df:              pd.DataFrame,
    close_scaler:    MinMaxScaler,
    step:            int  = 1,
    gate_on_signals: bool = False,
) -> pd.DataFrame:
    """
    Walk-forward backtest.

    Sentiment note
    --------------
    If sentiment columns are present in df (because --sentiment was passed),
    they are already in BOOL_COLS and flow through naturally as part of
    bool_window at each step. No special handling needed inside the loop —
    the pre-computation in load_and_prepare_data already ensured each bar's
    sentiment score only uses articles published before that bar.

    The only backtest-specific consideration is the close_scaler: it is
    loaded from the training checkpoint (fitted on training data), NOT
    re-fitted on the backtest window. Re-fitting would be a form of lookahead
    because the scaler would have seen future prices.
    """
    BOOL_COLS  = _get_bool_cols(df)
    close_vals = df["Close"].values.astype(float)
    bool_vals  = df[BOOL_COLS].values.astype(float)
    dates      = df.index

    results   = []
    min_start = WINDOW_SIZE + 1
    max_start = len(df) - PRED_STEPS
    total     = len(range(min_start, max_start, step))
    skipped   = 0

    mode_label = "gated" if gate_on_signals else "always predict"
    print(f"[backtest] {total} windows  |  step={step}  |  mode={mode_label}")

    # Log which sentiment bars are active across the full dataset
    sentiment_cols = [c for c in BOOL_COLS if c.startswith("news_")]
    if sentiment_cols:
        for col in sentiment_cols:
            rate = bool_vals[:, BOOL_COLS.index(col)].mean()
            print(f"[sentiment] {col:<35} firing rate: {rate:.3f}")

    for idx, i in enumerate(range(min_start, max_start, step)):

        close_window  = close_vals[i - WINDOW_SIZE : i]
        bool_window   = bool_vals [i - WINDOW_SIZE : i]
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
                # Surface which sentiment signals were active at entry
                **{col: bool(bool_window[-1, BOOL_COLS.index(col)])
                   for col in sentiment_cols},
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
            # Surface which sentiment signals were active at entry
            **{col: bool(bool_window[-1, BOOL_COLS.index(col)])
               for col in sentiment_cols},
        })

        if (idx + 1) % 50 == 0:
            print(f"  … {idx + 1}/{total} done")

    if gate_on_signals:
        print(f"[backtest] {skipped}/{total} windows skipped (no signal)")
    print(f"[backtest] Complete — {len(results)} windows recorded.\n")
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  METRICS  (+ sentiment breakdown)
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
        print("[warn] No windows were predicted.")
        return {}

    gated_count = int(results["gated_out"].sum())

    w = 60
    print("\n" + "═" * w)
    print("  LSTM WALK-FORWARD BACKTEST  —  PERFORMANCE REPORT")
    print("═" * w)
    print(f"\n  Dataset  : {TICKER} · {PERIOD} · {INTERVAL} bars")
    print(f"  Windows  : {total}  |  Predicted: {len(predicted)}  |  Gated: {gated_count}")
    print(f"  Look-back: {WINDOW_SIZE}  |  Horizon: {PRED_STEPS}  |  Threshold: ±{THRESHOLD}%")

    print(f"\n  ── Signal Distribution {'─' * (w - 25)}")
    for lbl in ["BUY", "HOLD", "SELL"]:
        print(f"    {lbl:<6}  predicted: {sig_counts.get(lbl,0):>5}   actual: {act_counts.get(lbl,0):>5}")

    print(f"\n  ── Accuracy {'─' * (w - 14)}")
    print(f"    Overall (3-class)      : {overall_acc:>6.1f} %")
    print(f"    Directional (no HOLD)  : {dir_acc:>6.1f} %")
    print(f"    BUY  accuracy          : {buy_acc:>6.1f} %")
    print(f"    SELL accuracy          : {sell_acc:>6.1f} %")
    print(f"    HOLD accuracy          : {hold_acc:>6.1f} %")

    print(f"\n  ── P&L (0.1% commission/side) {'─' * (w - 32)}")
    print(f"    Trades taken      : {len(traded):>6}")
    print(f"    Win rate          : {win_rate:>6.1f} %")
    print(f"    Avg win           : {avg_win:>+6.3f} %")
    print(f"    Avg loss          : {avg_loss:>+6.3f} %")
    print(f"    Profit factor     : {profit_fac:>7.3f}")
    print(f"    Avg P&L per trade : {avg_pnl:>+6.3f} %")
    print(f"    Total P&L (sum)   : {total_pnl:>+6.2f} %")
    print(f"    Cumulative return : {cum_return:>+6.2f} %")
    print(f"    Max drawdown      : {max_dd:>+6.2f} %")

    # ── Sentiment breakdown (only shown when sentiment cols exist) ────────────
    sentiment_cols = [c for c in results.columns if c.startswith("news_")]
    if sentiment_cols and len(traded) > 0:
        print(f"\n  ── Sentiment Breakdown {'─' * (w - 25)}")
        print(f"    Win rate when news_bull active     : ", end="")
        if "news_bull" in results.columns:
            bull_trades = traded[traded["news_bull"] == True]
            if len(bull_trades):
                bull_wr = (bull_trades["pnl_pct"] > 0).mean() * 100
                print(f"{bull_wr:.1f}%  ({len(bull_trades)} trades)")
            else:
                print("no trades")
        print(f"    Win rate when news_bear active     : ", end="")
        if "news_bear" in results.columns:
            bear_trades = traded[traded["news_bear"] == True]
            if len(bear_trades):
                bear_wr = (bear_trades["pnl_pct"] > 0).mean() * 100
                print(f"{bear_wr:.1f}%  ({len(bear_trades)} trades)")
            else:
                print("no trades")
        print(f"    Win rate with no sentiment signal  : ", end="")
        no_sent = traded[
            (traded.get("news_bull", pd.Series(False, index=traded.index)) == False) &
            (traded.get("news_bear", pd.Series(False, index=traded.index)) == False)
        ]
        if len(no_sent):
            print(f"{(no_sent['pnl_pct'] > 0).mean()*100:.1f}%  ({len(no_sent)} trades)")
        else:
            print("no trades")

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

def main():
    gate = GATE_ON_SIGNALS

    df        = load_and_prepare_data(use_sentiment=USE_SENTIMENT)
    bool_cols = _get_bool_cols(df)   # resolve after df is built

    print(f"[model] Loading from {MODEL_PATH} …")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    # ── Validate feature alignment ────────────────────────────────────────────
    ckpt_bool_cols = ckpt.get("bool_cols", None)
    if ckpt_bool_cols is not None and ckpt_bool_cols != bool_cols:
        missing  = set(ckpt_bool_cols) - set(bool_cols)
        extra    = set(bool_cols) - set(ckpt_bool_cols)
        msg = (
            f"\n[error] FEATURES mismatch between checkpoint and current feature_engineering.py\n"
            f"  In checkpoint but not in current FEATURES : {missing or 'none'}\n"
            f"  In current FEATURES but not in checkpoint : {extra or 'none'}\n"
            f"\n  → If you added sentiment features after training, retrain the model first.\n"
            f"  → Or remove the new features from FEATURES to match the checkpoint."
        )
        raise ValueError(msg)

    n_bool       = ckpt["n_bool_features"]
    hidden_size  = ckpt["hidden_size"]
    close_scaler = ckpt["close_scaler"]

    model = StockPriceLSTMNetwork(
        n_bool_features = n_bool,
        hidden_size     = hidden_size,
        output_size     = 1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[model] Loaded. n_bool={n_bool}, hidden={hidden_size}\n")

    results = run_walk_forward_backtest(
        model, df, close_scaler,
        step            = STEP,
        gate_on_signals = gate,
    )
    metrics = compute_and_print_metrics(results)

    results.to_csv(SAVE_RESULTS, index=False)
    print(f"[output] Results saved → {SAVE_RESULTS}")
    return results, metrics


if __name__ == "__main__":
    main()