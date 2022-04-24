import torch
from torch import nn


class MultiInputCNN(nn.Module):
    def __init__(self, conv_filters, fc_node_count, kernel_size, mers=3):
        super(MultiInputCNN, self).__init__()
        length = 36 - mers + 1
        padded_length = length + 2  # + 2 for padding
        in_channels = 4 ** mers
        self.conv_layer = nn.Sequential(
            nn.ConstantPad1d(1, .25),
            nn.Conv1d(in_channels, conv_filters, kernel_size),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        fc_node_count = fc_node_count
        self.dense_layer = nn.Sequential(
            nn.Linear((padded_length - kernel_size + 1)*conv_filters + in_channels*length,
                      fc_node_count),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(fc_node_count, 128),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(128, 1),
        )

    def forward(self, X):
        X1 = self.conv_layer(X)
        X1 = self.flatten(X1)
        X = self.flatten(X)
        X = torch.cat((X, X1), 1)
        X = self.dense_layer(X)
        return X


class ThreeLayerCNN(nn.Module):
    def __init__(self, conv_filters, conv2_filters, conv3_filters, fc_node_count, kernel_size,
                 kernel2_size, kernel3_size, mers=3):
        super(ThreeLayerCNN, self).__init__()
        padded_length = 36 - mers + 2 + 1  # + 2 for padding
        in_channels = 4 ** mers
        self.conv_layer = nn.Sequential(
            nn.ConstantPad1d(1, .25),
            nn.Conv1d(in_channels, conv_filters, kernel_size),
            nn.ReLU(),
        )
        self.conv2_layer = nn.Sequential(
            nn.Conv1d(conv_filters, conv2_filters, kernel2_size),
            nn.ReLU(),
        )

        self.conv3_layer = nn.Sequential(
            nn.Conv1d(conv2_filters, conv3_filters, kernel3_size),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        fc_node_count = fc_node_count
        self.dense_layer = nn.Sequential(
            nn.Linear((padded_length - kernel_size + 1 - kernel2_size + 1 - kernel3_size + 1)
                      * conv3_filters + 2, fc_node_count),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(fc_node_count, 128),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(128, 1),
        )

    def forward(self, ets1_score, runx1_score, X):
        X = self.conv_layer(X)
        X = self.conv2_layer(X)
        X = self.conv3_layer(X)
        X = self.flatten(X)
        X = torch.cat((X, ets1_score, runx1_score), 1)
        X = self.dense_layer(X)
        return X


class NLayerCNN(nn.Module):
    def __init__(self, conv_filters, fc_node_count, kernel_sizes, extra_feature_count, mers=3):
        super(NLayerCNN, self).__init__()
        padded_length = 36 - mers + 2 + 1  # + 2 for padding
        in_channels = 4 ** mers

        self.conv_layers = [
            nn.Sequential(
                nn.ConstantPad1d(1, .25),
                nn.Conv1d(in_channels, conv_filters[0], kernel_sizes[0]),
                nn.ReLU())
        ]

        for i in range(1, len(conv_filters)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(conv_filters[i-1], conv_filters[i], kernel_sizes[i]),
                nn.ReLU(),
            ))

        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.flatten = nn.Flatten()

        fc_node_count = fc_node_count
        fc_input_size = (extra_feature_count + conv_filters[-1]
                         * (padded_length + sum(1 - kernel_size for kernel_size in kernel_sizes)))

        self.dense_layer = nn.Sequential(
            nn.Linear(fc_input_size, fc_node_count),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(fc_node_count, 128),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(128, 1),
        )

    def forward(self, X, *extra_features):
        for conv_layer in self.conv_layers:
            X = conv_layer(X)
        X = self.flatten(X)
        # print(f"X.dims: {X.dim()}")
        # print(f"extra_features: {extra_features[1].shape}, {extra_features[2].shape}")
        if len(extra_features):
            X = torch.cat((X, *extra_features), 1)
        # print(f"concatX: {X.shape}")
        X = self.dense_layer(X)
        return X


class NLayerDilatedCNN(nn.Module):
    def __init__(self, conv_filters, fc_node_count, kernel_sizes, extra_feature_count, mers=3):
        super(NLayerDilatedCNN, self).__init__()
        dilation = 3
        padded_length = 36 - mers + 2 + 1  # + 2 for padding
        in_channels = 4 ** mers

        self.conv_layers = [
            nn.Sequential(
                nn.ConstantPad1d(1, .25),
                nn.Conv1d(in_channels, conv_filters[0], kernel_sizes[0], dilation=dilation),
                nn.ReLU())
        ]

        for i in range(1, len(conv_filters)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(conv_filters[i - 1], conv_filters[i], kernel_sizes[i], dilation=dilation),
                nn.ReLU(),
            ))

        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.flatten = nn.Flatten()

        fc_node_count = fc_node_count
        fc_input_size = (extra_feature_count + conv_filters[-1]
                         * (padded_length - sum((kernel_size-1)*dilation
                                                for kernel_size in kernel_sizes)))

        self.dense_layer = nn.Sequential(
            nn.Linear(fc_input_size, fc_node_count),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(fc_node_count, 128),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(128, 1),
        )

    def forward(self, X, *extra_features):
        for conv_layer in self.conv_layers:
            X = conv_layer(X)
        X = self.flatten(X)
        # print(f"X.dims: {X.dim()}")
        # print(f"extra_features: {extra_features[1].shape}, {extra_features[2].shape}")
        if len(extra_features):
            X = torch.cat((X, *extra_features), 1)
        # print(f"concatX: {X.shape}")
        X = self.dense_layer(X)
        return X


