import torch
import torch.nn as nn

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon, output_size=None):
        super(MultivariateLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.output_size = output_size if output_size is not None else input_size
        self.horizon = horizon
        self.linear = nn.Linear(hidden_size, horizon * self.output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        _, (hn, _) = self.lstm(x)
        # Use last layer's final hidden state
        out = hn[-1]
        out = self.dropout(out)
        out = self.linear(out)
        out = out.view(-1, self.horizon, self.output_size)
        return out