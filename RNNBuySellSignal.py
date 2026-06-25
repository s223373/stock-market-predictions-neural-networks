import yfinance as yf
import pandas as pd
import numpy as np
import torch
import datetime
from sklearn.preprocessing import MinMaxScaler
from news_sentiment import SentimentPipeline
from feature_engineering import build_features, FEATURES
from model import StockPriceLSTMNetwork, DirectionalLoss, prepare_price_input, StockPriceLSTMNetworkDualStream

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TICKER      = "SPY"
WINDOW_SIZE = 14
EPOCHS      = 200
HIDDEN_SIZE = 64
LR          = 0.001

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

pipe = SentimentPipeline(TICKER, days_back=35)

df = yf.download(TICKER, period="30d", interval="5m", progress=False)
df.columns = df.columns.get_level_values(0)
df.index   = pd.to_datetime(df.index)
df         = build_features(df, period="30d", interval="5m")
df         = pipe.add_sentiment_features(df)

assert FEATURES[0] == "Close", "FEATURES[0] must be 'Close'."

full      = df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
bool_cols = [f for f in FEATURES if f != "Close"]

close_vals = full["Close"].values.astype(np.float64)
bool_vals  = full[bool_cols].values.astype(np.float32)

# Sanity-check the raw arrays before we do anything else
assert not np.isnan(close_vals).any(), "NaN in close_vals after dropna"
assert not np.isinf(close_vals).any(), "Inf in close_vals"
assert not np.isnan(bool_vals).any(),  "NaN in bool_vals after dropna"
print(f"[data] {len(full)} clean bars  |  {len(bool_cols)} bool features")
print(f"[data] Close range: [{close_vals.min():.4f}, {close_vals.max():.4f}]")
print(f"[data] Bool  range: [{bool_vals.min():.1f}, {bool_vals.max():.1f}]")

close_scaler = MinMaxScaler(feature_range=(-1, 1))
close_norm   = close_scaler.fit_transform(close_vals.reshape(-1, 1)).flatten().astype(np.float32)

assert not np.isnan(close_norm).any(), "NaN in close_norm after scaling"
print(f"[data] Scaled close range: [{close_norm.min():.4f}, {close_norm.max():.4f}]")

close_tensor = torch.from_numpy(close_norm)   # float32
bool_tensor  = torch.from_numpy(bool_vals)    # float32
n_bool       = len(bool_cols)

# ─────────────────────────────────────────────────────────────────────────────
# WINDOWED DATASET
# ─────────────────────────────────────────────────────────────────────────────

def generate_windows(close_seq, bool_seq, window_size):
    windows = []
    for i in range(len(close_seq) - window_size):
        close_window = close_seq[i : i + window_size]       # (W,)
        bool_window  = bool_seq [i : i + window_size - 1]   # (W-1, n_bool)
        price_input  = prepare_price_input(close_window)     # (W-1, 1)
        label        = close_seq[i + window_size]            # scalar tensor
        windows.append((price_input, bool_window, label))
    return windows

train_data = generate_windows(close_tensor, bool_tensor, WINDOW_SIZE)
print(f"[data] {len(train_data)} training windows")

# Pre-flight: verify all windows are clean before training starts
print("[preflight] Checking all windows for NaN/Inf …")
for i, (price_input, bool_input, label) in enumerate(train_data):
    if torch.isnan(price_input).any() or torch.isinf(price_input).any():
        raise ValueError(f"Window {i}: NaN/Inf in price_input  raw={close_tensor[i:i+WINDOW_SIZE].tolist()}")
    if torch.isnan(bool_input).any() or torch.isinf(bool_input).any():
        raise ValueError(f"Window {i}: NaN/Inf in bool_input")
    if torch.isnan(label) or torch.isinf(label):
        raise ValueError(f"Window {i}: NaN/Inf in label  value={label.item()}")
print("[preflight] All windows clean ✓")

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

best_loss  = float("inf")
best_state = None

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    nan_hit    = False

    for i, (price_input, bool_input, y_train) in enumerate(train_data):
        optimizer.zero_grad()

        price_in = price_input.unsqueeze(0)    # (1, W-1, 1)
        bool_in  = bool_input.unsqueeze(0)     # (1, W-1, n_bool)
        y_target = y_train.unsqueeze(0)        # (1,)

        y_pred = model(price_in, bool_in)      # (1, 1)

        # ── Inline NaN detection — fires on first bad window ─────────────────
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            print(f"\n[NaN] Epoch {epoch+1}  window {i}: NaN/Inf in y_pred={y_pred.item():.6f}")
            print(f"      price_in  NaN={torch.isnan(price_in).any()}  Inf={torch.isinf(price_in).any()}")
            print(f"      bool_in   NaN={torch.isnan(bool_in).any()}   Inf={torch.isinf(bool_in).any()}")
            print(f"      y_target  = {y_target.item():.6f}")
            nan_hit = True
            break

        loss = criterion(y_pred.squeeze(-1), y_target)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[NaN] Epoch {epoch+1}  window {i}: NaN/Inf loss={loss.item()}")
            print(f"      y_pred   = {y_pred.item():.6f}")
            print(f"      y_target = {y_target.item():.6f}")
            nan_hit = True
            break

        loss.backward()

        # Check gradients before clipping
        bad_grads = [
            (n, p.grad.abs().max().item())
            for n, p in model.named_parameters()
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
        ]
        if bad_grads:
            print(f"\n[NaN] Epoch {epoch+1}  window {i}: NaN/Inf gradients in {[n for n,_ in bad_grads]}")
            nan_hit = True
            break

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    if nan_hit:
        print("Training halted — fix the NaN source above before continuing.")
        break

    avg_loss = epoch_loss / len(train_data)

    # Track best checkpoint
    if avg_loss < best_loss:
        best_loss  = avg_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3}  Avg Loss: {avg_loss:.8f}  Best: {best_loss:.8f}")

else:
    # ─────────────────────────────────────────────────────────────────────────
    # SAVE  (only reached if training completed without NaN)
    # Saves the best checkpoint (lowest avg loss), not the final epoch.
    # alpaca_trader.py loads this same dict — keep keys in sync.
    # ─────────────────────────────────────────────────────────────────────────
    now       = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"StockPriceLSTMNetwork_{now}.pt"
    torch.save({
        "model_state_dict": best_state,   # best epoch, not last
        "n_bool_features":  n_bool,
        "hidden_size":      HIDDEN_SIZE,
        "bool_cols":        bool_cols,
        "close_scaler":     close_scaler,
        "window_size":      WINDOW_SIZE,
    }, save_path)
    print(f"\nModel saved → {save_path}")
    print(f"Best avg loss: {best_loss:.8f}")