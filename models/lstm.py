import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, num_classes: int = 1):
        super().__init__()
        lstm_dropout = 0.5 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq, feat)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)
