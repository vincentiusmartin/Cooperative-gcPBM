import torch
from torch import nn
from skorch import NeuralNetRegressor


class NLayerCNN(nn.Module):
    def __init__(self, conv_filters, fc_node_count, kernel_sizes, include_affinities=False,
                 pool=None, mers=3):
        super(NLayerCNN, self).__init__()
        self.include_affinities = include_affinities

        padded_length = 36 - mers + 2 + 1  # + 2 for padding
        in_channels = 4 ** mers

        if type(conv_filters) == int:
            conv_filters = [conv_filters] * len(kernel_sizes)

        self.conv_layers = [
            nn.Sequential(
                nn.ConstantPad1d(1, 1/(4**mers)),
                nn.Conv1d(in_channels, conv_filters[0], kernel_sizes[0]),
                nn.ReLU(),
            )
        ]

        for i in range(1, len(conv_filters)):
            params = [
                nn.Conv1d(conv_filters[i-1], conv_filters[i], kernel_sizes[i]),
                nn.ReLU()
            ]

            if pool == "max":
                params.append(nn.MaxPool1d(2))

            self.conv_layers.append(nn.Sequential(*params))

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.flatten = nn.Flatten()

        # compute remaining horizontal positions
        if pool is not None:
            one_d_length = (padded_length - kernel_sizes[0] + 1)

            for kernel_size in kernel_sizes[1:]:
                one_d_length = (one_d_length - kernel_size + 1) // 2
        else:
            one_d_length = padded_length + sum(1 - kernel_size for kernel_size in kernel_sizes)

        fc_input_size = conv_filters[-1] * one_d_length

        if include_affinities:
            fc_input_size += 2

        self.dense_layer = nn.Sequential(
            nn.Linear(fc_input_size, fc_node_count),
            nn.ReLU(),
            nn.Linear(fc_node_count, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, sequences, site1_scores=None, site2_scores=None):
        X = sequences.transpose(1, 2)
        for conv_layer in self.conv_layers:
            X = conv_layer(X)

        X = self.flatten(X)

        if self.include_affinities:
            X = torch.cat((X, site1_scores, site2_scores), 1).float()

        X = self.dense_layer(X)
        return X


class SkorchNeuralNetRegressor(NeuralNetRegressor):
    def fit(self, X, y, **kwargs):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return super().fit(X, y, **kwargs)
