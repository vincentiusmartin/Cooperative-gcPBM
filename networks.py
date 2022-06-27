import torch
from torch import nn


class NLayerCNN(nn.Module):
    def __init__(self, conv_filters, fc_node_count, kernel_sizes, extra_feature_count, pool=None,
                 mers=3):
        super(NLayerCNN, self).__init__()
        padded_length = 36 - mers + 2 + 1  # + 2 for padding
        in_channels = 4 ** mers

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

            if pool == "avg":
                params.append(nn.AvgPool1d(2))

            self.conv_layers.append(nn.Sequential(*params))

        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.flatten = nn.Flatten()

        fc_node_count = fc_node_count

        # compute remaining horizontal positions
        if pool is not None:
            one_d_length = (padded_length - kernel_sizes[0] + 1)

            for kernel_size in kernel_sizes[1:]:
                one_d_length = (one_d_length - kernel_size + 1) // 2
        else:
            one_d_length = padded_length + sum(1 - kernel_size for kernel_size in kernel_sizes)

        fc_input_size = extra_feature_count + conv_filters[-1] * one_d_length

        self.dense_layer = nn.Sequential(
            nn.Linear(fc_input_size, fc_node_count),
            nn.ReLU(),
            nn.Linear(fc_node_count, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, X, *extra_features):
        for conv_layer in self.conv_layers:
            X = conv_layer(X)

        X = self.flatten(X)

        if len(extra_features):
            X = torch.cat((X, *extra_features), 1)

        X = self.dense_layer(X)
        return X
