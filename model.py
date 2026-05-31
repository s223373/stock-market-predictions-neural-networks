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


class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.huber = nn.HuberLoss()

    def forward(self, pred, target):
        huber_loss      = self.huber(pred, target)
        wrong_direction = (torch.sign(pred) != torch.sign(target)).float()
        dir_loss        = (wrong_direction * torch.abs(pred - target)).mean()
        return self.alpha * huber_loss + (1 - self.alpha) * dir_loss
