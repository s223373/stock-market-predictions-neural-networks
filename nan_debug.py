"""
nan_debug.py
============
Run this before training to pinpoint exactly where NaN is coming from.
    python nan_debug.py
"""

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import build_features, FEATURES
from model import StockPriceLSTMNetwork, DirectionalLoss, prepare_price_input, StockPriceLSTMNetworkDualStream

TICKER      = "AAPL"
WINDOW_SIZE = 14
HIDDEN_SIZE = 64

print("=" * 60)
print("STEP 1 — Raw download")
print("=" * 60)
df = yf.download(TICKER, period="30d", interval="5m", progress=False)
df.columns = df.columns.get_level_values(0)
df.index   = pd.to_datetime(df.index)
print(f"Shape: {df.shape}")
print(f"NaN in OHLCV:\n{df[['Open','High','Low','Close','Volume']].isna().sum()}")
print(f"Inf in Close: {np.isinf(df['Close'].values).sum()}")
print(f"Zero/neg Close: {(df['Close'].values <= 0).sum()}")

print("\n" + "=" * 60)
print("STEP 2 — After build_features (before dropna)")
print("=" * 60)
df = build_features(df)
for col in FEATURES:
    if col not in df.columns:
        print(f"  MISSING COLUMN: {col}")
        continue
    n_nan = df[col].isna().sum()
    n_inf = np.isinf(df[col].replace([np.nan], 0).values).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  {col:<35}  NaN={n_nan}  Inf={n_inf}")

print("\n" + "=" * 60)
print("STEP 3 — After replace inf + dropna")
print("=" * 60)
full = df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
print(f"Rows remaining: {len(full)}  (dropped {len(df) - len(full)})")
print(f"Any NaN left:   {full.isna().any().any()}")
print(f"Any Inf left:   {np.isinf(full.values).any()}")

bool_cols  = [f for f in FEATURES if f != "Close"]
close_vals = full["Close"].values.astype(float)
bool_vals  = full[bool_cols].values.astype(float)

print("\n" + "=" * 60)
print("STEP 4 — MinMaxScaler on Close")
print("=" * 60)
close_scaler = MinMaxScaler(feature_range=(-1, 1))
close_norm   = close_scaler.fit_transform(close_vals.reshape(-1, 1)).flatten()
print(f"Close range before scaling: [{close_vals.min():.4f}, {close_vals.max():.4f}]")
print(f"Close range after scaling:  [{close_norm.min():.4f}, {close_norm.max():.4f}]")
print(f"NaN after scaling: {np.isnan(close_norm).sum()}")
print(f"Inf after scaling: {np.isinf(close_norm).sum()}")

close_tensor = torch.FloatTensor(close_norm)
bool_tensor  = torch.FloatTensor(bool_vals)

print("\n" + "=" * 60)
print("STEP 5 — prepare_price_input on each window")
print("=" * 60)
bad_windows = []
for i in range(len(close_tensor) - WINDOW_SIZE):
    window = close_tensor[i : i + WINDOW_SIZE]
    p = prepare_price_input(window)
    if torch.isnan(p).any() or torch.isinf(p).any():
        bad_windows.append((i, window.tolist()))

if bad_windows:
    print(f"  BAD windows found: {len(bad_windows)}")
    for idx, vals in bad_windows[:5]:
        print(f"    window {idx}: close vals = {[round(v,6) for v in vals]}")
else:
    print(f"  All {len(close_tensor) - WINDOW_SIZE} windows are clean ✓")

print("\n" + "=" * 60)
print("STEP 6 — Model forward pass (first 20 windows)")
print("=" * 60)
n_bool = len(bool_cols)
model  = StockPriceLSTMNetworkDualStream(n_bool_features=n_bool, hidden_size=HIDDEN_SIZE, output_size=1)
model.eval()

first_nan_forward = None
with torch.no_grad():
    for i in range(min(20, len(close_tensor) - WINDOW_SIZE)):
        price_in = prepare_price_input(close_tensor[i:i+WINDOW_SIZE]).unsqueeze(0)
        bool_in  = bool_tensor[i:i+WINDOW_SIZE-1].unsqueeze(0)
        label    = close_tensor[i + WINDOW_SIZE]

        out = model(price_in, bool_in)

        if torch.isnan(out).any() or torch.isinf(out).any():
            first_nan_forward = i
            print(f"  NaN/Inf in model output at window {i}")
            print(f"    price_in has NaN: {torch.isnan(price_in).any().item()}")
            print(f"    price_in has Inf: {torch.isinf(price_in).any().item()}")
            print(f"    bool_in  has NaN: {torch.isnan(bool_in).any().item()}")
            print(f"    output: {out}")
            break

if first_nan_forward is None:
    print("  First 20 forward passes are clean ✓")

print("\n" + "=" * 60)
print("STEP 7 — Loss computation (first 20 windows)")
print("=" * 60)
criterion = DirectionalLoss(alpha=0.7, temp=5.0)
model.train()

first_nan_loss = None
for i in range(min(20, len(close_tensor) - WINDOW_SIZE)):
    price_in = prepare_price_input(close_tensor[i:i+WINDOW_SIZE]).unsqueeze(0)
    bool_in  = bool_tensor[i:i+WINDOW_SIZE-1].unsqueeze(0)
    label    = close_tensor[i + WINDOW_SIZE].unsqueeze(0)

    out  = model(price_in, bool_in).squeeze(-1)
    loss = criterion(out, label)

    if torch.isnan(loss) or torch.isinf(loss):
        first_nan_loss = i
        print(f"  NaN/Inf loss at window {i}")
        print(f"    pred  : {out.item():.6f}")
        print(f"    target: {label.item():.6f}")
        print(f"    huber : {torch.nn.HuberLoss()(out, label).item()}")
        print(f"    tanh_pred  : {torch.tanh(5.0 * out).item():.6f}")
        print(f"    tanh_target: {torch.tanh(5.0 * label).item():.6f}")
        break

if first_nan_loss is None:
    print("  First 20 loss computations are clean ✓")

print("\n" + "=" * 60)
print("STEP 8 — Gradient check (5 backward passes)")
print("=" * 60)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(5):
    optimizer.zero_grad()
    price_in = prepare_price_input(close_tensor[i:i+WINDOW_SIZE]).unsqueeze(0)
    bool_in  = bool_tensor[i:i+WINDOW_SIZE-1].unsqueeze(0)
    label    = close_tensor[i + WINDOW_SIZE].unsqueeze(0)

    out  = model(price_in, bool_in).squeeze(-1)
    loss = criterion(out, label)
    loss.backward()

    # Check for NaN/Inf gradients before clipping
    nan_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_grads.append(name)

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"  window {i}: loss={loss.item():.6f}  grad_norm={total_norm:.4f}"
          + (f"  NaN grads: {nan_grads}" if nan_grads else "  grads OK ✓"))
    optimizer.step()

print("\n" + "=" * 60)
print("STEP 9 — Bool feature firing rate")
print("=" * 60)
firing = bool_vals.mean(axis=0)
for col, rate in zip(bool_cols, firing):
    status = "  ← NEVER FIRES" if rate == 0 else ("  ← ALWAYS ON" if rate == 1 else "")
    print(f"  {col:<40} {rate:.3f}{status}")

print("\nDone. Share the output above to identify the NaN source.")
