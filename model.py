"""
model.py
========
LSTM model definition. Import StockPriceLSTMNetwork from here
in the training script, backtest, and scheduler.
"""

import torch
import torch.nn as nn


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


"""
model.py
========
LSTM model definition. Import StockPriceLSTMNetwork from here
in the training script, backtest, and scheduler.

Architecture overview
---------------------
StockPriceLSTMNetwork uses a dual-stream design:

  Stream 1 — Price LSTM
    Input : log-returns of Close, z-scored over the window (shape: seq_len-1, 1)
    Module: single LSTM → TemporalAttention pooling

  Stream 2 — Boolean signal LSTM
    Input : boolean features linearly embedded → GELU → LayerNorm before the LSTM
    Module: single LSTM → TemporalAttention pooling

  Fusion : concat both stream outputs → LayerNorm → Linear → GELU → Dropout → output

The two streams are kept separate until fusion so the model can learn
independent temporal patterns from the continuous price series and the
discrete signal flags without conflating their scales.

Utilities
---------
prepare_price_input(close_seq)   Convert a raw Close tensor to z-scored log-returns.
                                 Call this before passing price data to the model.

DirectionalLoss                  Differentiable combination of HuberLoss and a soft
                                 directional agreement penalty (tanh-based, no dead
                                 gradients from torch.sign).
"""



# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def prepare_price_input(close_seq: torch.Tensor) -> torch.Tensor:
    """
    Convert a raw Close price sequence to z-scored log-returns.

    Parameters
    ----------
    close_seq : Tensor of shape (seq_len,) or (batch, seq_len)
        Raw closing prices.

    Returns
    -------
    Tensor of shape (seq_len-1, 1) or (batch, seq_len-1, 1)
        Stationary, scale-invariant price representation ready for Stream 1.

    Why
    ---
    Raw price is non-stationary and regime-dependent — $150 in 2020 carries
    a different distribution to $150 in 2024. Log-returns are stationary and
    z-scoring over the input window makes the representation scale-invariant,
    which is exactly what LSTMs need to generalise across price levels.
    """
    if close_seq.dim() == 1:
        log_ret = torch.log(close_seq[1:] / close_seq[:-1].clamp(min=1e-9))
        mu  = log_ret.mean()
        std = log_ret.std().clamp(min=1e-6)
        return ((log_ret - mu) / std).unsqueeze(-1)          # (seq_len-1, 1)
    else:
        # batched: (batch, seq_len)
        log_ret = torch.log(close_seq[:, 1:] / close_seq[:, :-1].clamp(min=1e-9))
        mu  = log_ret.mean(dim=1, keepdim=True)
        std = log_ret.std(dim=1,  keepdim=True).clamp(min=1e-6)
        return ((log_ret - mu) / std).unsqueeze(-1)          # (batch, seq_len-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Soft attention pooling over the LSTM time dimension.

    Rather than discarding all but the last hidden state, attention lets
    the model learn which timesteps in the lookback window were most
    informative — e.g. the exact sweep candle several bars back.

    Input  : (batch, seq_len, hidden_size)
    Output : (batch, hidden_size)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores  = self.attn(lstm_out).squeeze(-1)            # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1) # (batch, seq_len, 1)
        return (lstm_out * weights).sum(dim=1)               # (batch, hidden_size)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class StockPriceLSTMNetworkDualStream(nn.Module):
    """
    Dual-stream LSTM for mixed continuous + boolean feature inputs.

    Parameters
    ----------
    n_bool_features : int
        Number of boolean signal columns (everything except Close).
    hidden_size : int
        Hidden dimension for the price LSTM. The boolean LSTM uses hidden_size // 2.
    output_size : int
        Number of output values (typically 1 for next-bar return prediction).
    num_layers : int
        Number of stacked LSTM layers in each stream.
    dropout : float
        Dropout probability applied in the fusion head and between LSTM layers.

    Usage
    -----
        price_input = prepare_price_input(batch_close)          # (B, T-1, 1)
        bool_input  = batch_features[:, 1:, 1:]                 # (B, T-1, n_bool)
        pred        = model(price_input, bool_input)
    """

    def __init__(
        self,
        n_bool_features: int,
        hidden_size:     int,
        output_size:     int,
        num_layers:      int = 1,
        dropout:         float = 0.3,
    ):
        super().__init__()
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        bool_embed_dim      = max(16, n_bool_features // 2)
        bool_hidden         = hidden_size // 2

        # ── Stream 1: price (log-returns) ────────────────────────────────────
        self.price_lstm = nn.LSTM(
            input_size  = 1,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1 if num_layers > 1 else 0.0,
        )
        self.price_attn = TemporalAttention(hidden_size)

        # ── Stream 2: boolean signals ─────────────────────────────────────────
        # Embed raw 0/1 values into a learned dense representation before the
        # LSTM. Booleans don't have magnitude — embedding lets the model learn
        # which combinations of signals matter before temporal processing.
        self.bool_embed = nn.Sequential(
            nn.Linear(n_bool_features, bool_embed_dim),
            nn.GELU(),
            nn.LayerNorm(bool_embed_dim),
        )
        self.bool_lstm = nn.LSTM(
            input_size  = bool_embed_dim,
            hidden_size = bool_hidden,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1 if num_layers > 1 else 0.0,
        )
        self.bool_attn = TemporalAttention(bool_hidden)

        # ── Fusion head ───────────────────────────────────────────────────────
        fused_dim = hidden_size + bool_hidden
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for lstm in (self.price_lstm, self.bool_lstm):
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Set forget-gate bias = 1 to encourage remembering early in training.
                    # LSTM bias layout per layer: [input | forget | cell | output]
                    # each gate occupies n // 4 elements.
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.bool_embed[0].weight)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, price_input: torch.Tensor, bool_input: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        price_input : Tensor (batch, seq_len, 1)
            Z-scored log-returns from prepare_price_input().
        bool_input  : Tensor (batch, seq_len, n_bool_features)
            Boolean feature columns as float32 (0.0 / 1.0).

        Returns
        -------
        Tensor (batch, output_size)
        """
        B = price_input.size(0)

        # Stream 1 — price
        h0_p = torch.zeros(self.num_layers, B, self.hidden_size,  device=price_input.device)
        c0_p = torch.zeros(self.num_layers, B, self.hidden_size,  device=price_input.device)
        price_out, _ = self.price_lstm(price_input, (h0_p, c0_p))  # (B, T, H)
        price_ctx    = self.price_attn(price_out)                   # (B, H)

        # Stream 2 — boolean signals
        bool_hidden = self.bool_lstm.hidden_size
        h0_b = torch.zeros(self.num_layers, B, bool_hidden, device=bool_input.device)
        c0_b = torch.zeros(self.num_layers, B, bool_hidden, device=bool_input.device)
        bool_emb     = self.bool_embed(bool_input)                  # (B, T, embed_dim)
        bool_out, _  = self.bool_lstm(bool_emb, (h0_b, c0_b))      # (B, T, H//2)
        bool_ctx     = self.bool_attn(bool_out)                     # (B, H//2)

        # Fusion
        fused = torch.cat([price_ctx, bool_ctx], dim=-1)            # (B, H + H//2)
        return self.fusion(fused)                                    # (B, output_size)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

class DirectionalLoss(nn.Module):
    """
    Differentiable combination of HuberLoss and a soft directional penalty.

    The original implementation used torch.sign() for directional agreement,
    which has zero gradient almost everywhere and contributes nothing to
    backprop. This version replaces it with tanh(temp * x), which approximates
    sign(x) as temp → ∞ while remaining fully differentiable throughout.

    Loss = alpha * Huber(pred, target)
         + (1 - alpha) * mean(1 - tanh(temp*pred) * tanh(temp*target))

    The directional term is 0 when signs agree perfectly and 2 at worst.

    Parameters
    ----------
    alpha : float
        Weight on the Huber magnitude loss (0–1). Default 0.7.
    temp  : float
        Sharpness of the soft sign. Higher = closer to hard sign but
        still differentiable. Default 5.0.
    """

    def __init__(self, alpha: float = 0.7, temp: float = 5.0):
        super().__init__()
        self.alpha = alpha
        self.temp  = temp
        self.huber = nn.HuberLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber_loss = self.huber(pred, target)

        # Soft directional agreement: +1 when signs match, -1 when opposite.
        # (1 - agreement) is 0 for perfect direction, 2 for wrong direction.
        soft_agreement = (
            torch.tanh(self.temp * pred) *
            torch.tanh(self.temp * target)
        )
        dir_loss = (1.0 - soft_agreement).mean()

        return self.alpha * huber_loss + (1.0 - self.alpha) * dir_loss