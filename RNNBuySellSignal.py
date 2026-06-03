import yfinance as yf
import pandas as pd
import numpy as np
import ta
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
from feature_engineering import build_features, FEATURES
from model import StockPriceLSTMNetwork, DirectionalLoss, prepare_price_input, StockPriceLSTMNetworkDualStream

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TICKER      = "AAPL"
WINDOW_SIZE = 14
EPOCHS      = 200
HIDDEN_SIZE = 64
LR          = 0.001

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

df = yf.download(TICKER, period="30d", interval="5m")
df.columns = df.columns.get_level_values(0)
df.index   = pd.to_datetime(df.index)
df         = build_features(df)

# FEATURES[0] must be "Close" — the dual-stream model treats it separately.
assert FEATURES[0] == "Close", "FEATURES[0] must be 'Close' for the dual-stream model."

full = df[FEATURES].dropna()
full = df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
date_index = full.index.tz_convert("US/Pacific")

# Split into price (Close) and boolean signal arrays
close_vals = full["Close"].values.astype(float)          # (N,)
bool_cols  = [f for f in FEATURES if f != "Close"]
bool_vals = full[bool_cols].values.astype(float)       # (N, n_bool)

n_bool = len(bool_cols)

# ── Normalise Close with MinMaxScaler so the label is still in (-1, 1) ──────
# Boolean features are already 0/1 — no scaling needed.
from sklearn.preprocessing import MinMaxScaler
close_scaler = MinMaxScaler(feature_range=(-1, 1))
close_norm   = close_scaler.fit_transform(close_vals.reshape(-1, 1)).flatten()  # (N,)

close_tensor = torch.FloatTensor(close_norm)   # (N,)
bool_tensor  = torch.FloatTensor(bool_vals)    # (N, n_bool)

# ─────────────────────────────────────────────────────────────────────────────
# WINDOWED DATASET
# ─────────────────────────────────────────────────────────────────────────────

def generate_windows(close_seq, bool_seq, window_size):
    """
    For each window of length `window_size`:
      - price_input : prepare_price_input on close[i : i+window_size]
                      → log-returns, z-scored, shape (window_size-1, 1)
      - bool_input  : bool signals for the same window, aligned to the
                      log-return length → shape (window_size-1, n_bool)
      - label       : normalised Close value at bar i+window_size (next bar)
    """
    windows = []
    L = len(close_seq)
    for i in range(L - window_size):
        close_window = close_seq[i : i + window_size]           # (W,)
        bool_window  = bool_seq[i : i + window_size - 1]        # (W-1, n_bool)

        price_input  = prepare_price_input(close_window)         # (W-1, 1)
        label        = close_seq[i + window_size]                # scalar
        windows.append((price_input, bool_window, label))
    return windows

train_data = generate_windows(close_tensor, bool_tensor, WINDOW_SIZE)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(101)

model     = StockPriceLSTMNetworkDualStream(n_bool_features=n_bool, hidden_size=HIDDEN_SIZE, output_size=1)
criterion = DirectionalLoss(alpha=0.7, temp=5.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for price_input, bool_input, y_train in train_data:
        optimizer.zero_grad()

        # Model expects (batch, seq_len, features) — add batch dim of 1
        price_in = price_input.unsqueeze(0)   # (1, W-1, 1)
        bool_in  = bool_input.unsqueeze(0)    # (1, W-1, n_bool)
        y_target = y_train.unsqueeze(0)       # (1,)

        y_pred = model(price_in, bool_in)     # (1, 1)
        loss   = criterion(y_pred.squeeze(-1), y_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_data)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1:3}  Avg Loss: {avg_loss:.8f}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"StockPriceLSTMNetwork_{now}.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "n_bool_features":  n_bool,
    "hidden_size":      HIDDEN_SIZE,
    "bool_cols":        bool_cols,
    "close_scaler":     close_scaler,
    "window_size":      WINDOW_SIZE,
}, save_path)
print(f"\nModel saved → {save_path}")