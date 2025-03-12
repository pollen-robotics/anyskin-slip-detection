import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, lstm_hidden_size: int, output_size: int, nlayers: int):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, nlayers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size, hidden_size)

        # Output layer
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial hidden state and cell state
        h0 = torch.zeros(self.nlayers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.nlayers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out_lstm, _ = self.lstm(x, (h0, c0))

        # Pass through fully connected layer
        out_fc = self.fc(out_lstm[:, -1, :])

        # Pass through output layer
        out: torch.Tensor = self.out(out_fc)

        return out
