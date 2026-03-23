"""LSTM model for stock price prediction."""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """2-layer LSTM with fully-connected head for regression.

    Input:  (batch, seq_len, n_features)
    Output: (batch,) predicted Close price
    """

    def __init__(self, n_features: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        last_hidden = self.layer_norm(last_hidden)
        out = self.fc(last_hidden).squeeze(-1)  # (batch,)
        return out
