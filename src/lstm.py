"""LSTM Network for slip detection."""

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """LSTM Network."""

    def __init__(self, input_size: int, hidden_size: int, lstm_hidden_size: int, output_size: int, nlayers: int):
        """Initialize the LSTM Network for sequence classification.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            lstm_hidden_size (int): The number of features in the LSTM hidden state.
            output_size (int): The number of features in the output.
            nlayers (int): The number of recurrent layers.
        """
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
        """Define the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
        """
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
