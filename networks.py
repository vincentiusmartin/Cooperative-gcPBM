from torch import nn


class CNN(nn.Module):
    def __init__(self, conv_filters, fc_node_count, kernel_size, mers=3):
        super(CNN, self).__init__()
        padded_length = 36 - mers + 2 + 1  # + 2 for padding
        in_channels = 4 ** mers
        conv_filters = conv_filters
        kernel_size = kernel_size
        self.conv_layer = nn.Sequential(
            nn.ConstantPad1d(1, .25),
            nn.Conv1d(in_channels, conv_filters, kernel_size),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        fc_node_count = fc_node_count
        self.dense_layer = nn.Sequential(
            nn.Linear((padded_length - kernel_size + 1)*conv_filters, fc_node_count),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(fc_node_count, 128),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(128, 1),
        )

    def forward(self, X):
        X = self.conv_layer(X)
        X = self.flatten(X)
        X = self.dense_layer(X)
        return X


class TwoLayerCNN(nn.Module):
    def __init__(self, conv_filters, fc_node_count, kernel_size, conv2_filters, kernel2_size,
                 mers=3):
        super(TwoLayerCNN, self).__init__()
        padded_length = 36 - mers + 2  # + 2 for padding
        in_channels = 4 ** mers
        conv_filters = conv_filters
        kernel_size = kernel_size
        self.conv_layer = nn.Sequential(
            nn.ConstantPad1d(1, .25),
            nn.Conv1d(in_channels, conv_filters, kernel_size),
            nn.ReLU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(conv_filters, conv2_filters, kernel2_size),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        fc_node_count = fc_node_count
        self.dense_layer = nn.Sequential(
            nn.Linear((padded_length - kernel_size + 1 - kernel2_size + 1)*conv2_filters,
                      fc_node_count),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(fc_node_count, 128),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(128, 1),
        )

    def forward(self, X):
        X = self.conv_layer(X)
        X = self.conv2_layer(X)
        X = self.flatten(X)
        X = self.dense_layer(X)
        return X
