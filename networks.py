import torch
from torch import nn
from skorch import NeuralNetRegressor


class NLayerCNN(nn.Module):
    """A convolutional neural network for which a number of parameters can be configured.
    Number of layers of convolution are determined by the number of elements in kernel_widths list
    argument.
    """
    def __init__(self, conv_filters, fc_node_count, kernel_widths, include_affinities=False,
                 pool=False, mers=3):
        """Initialize the convolutional neural network.

        Args:
        conv_filters: number of convolutional filters. Can be a single integer, in which case all
        layers will have same number of filters. If a list, it specifies the number of filters at
        each layer.
        fc_node_count: This specifies the number of nodes in the first fully-connected layer.
        kernel_widths: This list specifies the size of the filters for the kernels at each layer.
        The size of the neural network is determined by the length of the kernel_widths argument.
        include_affinities: whether or not to include affinities.
        pool: whether to apply max pooling after all layers (except first).
        mers: number of mers.
        """
        super(NLayerCNN, self).__init__()
        self.include_affinities = include_affinities

        padded_length = 36 - mers + 2 + 1  # + 2 for padding
        in_channels = 4 ** mers

        if type(conv_filters) == int:
            conv_filters = [conv_filters] * len(kernel_widths)

        self.conv_layers = [
            nn.Sequential(
                nn.ConstantPad1d(1, 1/(4**mers)),
                nn.Conv1d(in_channels, conv_filters[0], kernel_widths[0]),
                nn.ReLU(),
            )
        ]

        for i in range(1, len(conv_filters)):
            params = [
                nn.Conv1d(conv_filters[i-1], conv_filters[i], kernel_widths[i]),
                nn.ReLU()
            ]

            if pool:
                params.append(nn.MaxPool1d(2))

            self.conv_layers.append(nn.Sequential(*params))

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.flatten = nn.Flatten()

        # compute remaining horizontal positions
        if pool:
            one_d_length = (padded_length - kernel_widths[0] + 1)

            for kernel_size in kernel_widths[1:]:
                one_d_length = (one_d_length - kernel_size + 1) // 2
        else:
            one_d_length = padded_length + sum(1 - kernel_size for kernel_size in kernel_widths)

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
    """Override 'fit' to re-expand to 2 dimensions, as skorch requires.
    This is necessary because TransformedTargetRegressor reduces target tensor to 1 dimension.
    """
    def __init__(self, **kwargs):
        kwargs["module"] = NLayerCNN
        super(SkorchNeuralNetRegressor, self).__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return super().fit(X, y, **kwargs)
