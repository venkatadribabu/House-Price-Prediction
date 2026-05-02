from __future__ import annotations

import torch
from torch import nn


class HousePriceLSTM(nn.Module):
    """
    LSTM + linear head for one-step-ahead scaled log-price.
    Single-layer baseline (no inter-layer dropout).
    """

    def __init__(
        self,
        lookback: int,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        residual_readout: bool = False,
    ):
        super().__init__()
        self.lookback = lookback
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.residual_readout = residual_readout
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = self.dropout(out[:, -1, :])
        delta = self.dense(last)
        if self.residual_readout:
            return delta + x[:, -1, 0:1]
        return delta


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
