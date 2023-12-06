# import torch
# import torch.nn as nn
#
# class AudioDenoisingRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(AudioDenoisingRNN, self).__init__()
#         self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # x shape: [batch, seq_length, features]
#         out, _ = self.rnn(x)
#         out = self.fc(out)
#         # Output shape: [batch, seq_length, output_size]
#         return out
#
# # Example initialization:
# # model = AudioDenoisingRNN(input_size=1025, hidden_size=128, num_layers=2, output_size=1025)

import torch
import torch.nn as nn


class AudioDenoisingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AudioDenoisingRNN, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch, seq_length, features]
        out, _ = self.rnn(x)
        out = self.fc(out)
        # Output shape: [batch, seq_length, output_size]
        return out


# Example initialization
# input_size = number of frequency bins in the spectrogram
# hidden_size = size of RNN hidden state
# num_layers = number of stacked RNN layers
# output_size = same as input_size (reconstructing the input spectrogram)
# model = AudioDenoisingRNN(input_size=1025, hidden_size=128, num_layers=2, output_size=1025)
