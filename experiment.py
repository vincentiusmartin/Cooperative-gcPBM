# import json
import matplotlib.pyplot as plt
from sklearn import metrics
import statistics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import get_train_and_validate_datasets


class NeuralNetwork(nn.Module):
    def __init__(self, conv_filters, kernel_size, fc_node_count):
        super(NeuralNetwork, self).__init__()
        padded_length = 35  # 36  # 38
        in_channels = 64  # 16  # 4
        conv_filters = conv_filters
        kernel_size = kernel_size
        self.conv_layer = nn.Sequential(
            nn.ConstantPad1d(1, .25),
            nn.Conv1d(in_channels, conv_filters, kernel_size),
            nn.ReLU(),
            # nn.MaxPool1d(4, 4)
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


def train(optimizer, loss_fn, model, dataloader):
    model.train()

    for X, y_true in dataloader:
        y_hat = model(X)
        y_hat = y_hat.flatten()
        loss = loss_fn(y_hat, y_true)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


# accumulate SS_residual and SS_total to compute R2 = 1 - SS_residual/SS_total

def get_r2(model, dataloader):
    y_hat_arr, y_true_arr = [], []
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            y_hat = model(X)
            y_hat = y_hat.flatten()
            y_hat_arr += list(y_hat)
            y_true_arr += list(y)

    return metrics.r2_score(y_true_arr, y_hat_arr)


# accumulate SS_residual and SS_total to compute R2 = 1 - SS_residual/SS_total

def validate(model, dataloader):
    y_hat_arr, y_true_arr = [], []
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            y_hat = model(X)
            y_hat = y_hat.flatten()
            y_hat_arr += list(y_hat)
            y_true_arr += list(y)

    return metrics.mean_squared_error(y_true_arr, y_hat_arr)


def plot_train_validate_accuracy(train_series, validate_series):
    epochs = len(train_series)
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(16, 6)

    ax1.plot(range(epochs), train_series, label="training")
    ax1.plot(range(epochs), validate_series, label="validation")

    ax1.axhline(y=0, color="grey", linestyle="--")
    ax1.axhline(y=max(validate_series), color="grey", linestyle="--")
    plt.xlabel("epochs")
    plt.ylabel("accuracy ($R^2$)")
    ax1.legend()
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # print(NeuralNetwork(128, 18, 256))

    # grid = [(conv_filters, kernel_size, fc_node_count)
    #         for conv_filters in [16, 32, 48, 64, 128, 256]
    #         for kernel_size in [6, 8, 10, 12, 18, 24, 28, 32]
    #         for fc_node_count in [128, 256]]
    grid = [(128, 18, 256)]

    batch_size = 128
    epochs = 50
    experiment = "ets1_ets1"

    results = {}

    for params in grid:
        max_validate_acc = []
        conv_filters, kernel_size, fc_node_count = params
        print(f"conv_filters: {conv_filters}, kernel_size: {kernel_size},"
              f" fc_node_count: {fc_node_count}")

        for i in range(10):
            net = NeuralNetwork(*params).to(device)

            train_data, validate_data = get_train_and_validate_datasets(experiment, mers=3)

            train_dataloader = DataLoader(train_data, batch_size=batch_size)
            validate_dataloader = DataLoader(validate_data, batch_size=batch_size)

            optimizer = torch.optim.Adam(net.parameters())
            loss_fn = nn.MSELoss()

            train_acc = []
            validate_acc = []

            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Training the model 1 time                           #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            for t in range(epochs):
                train(optimizer, loss_fn, net, train_dataloader)
                train_acc.append(get_r2(net, train_dataloader))
                validate_acc.append(get_r2(net, validate_dataloader))

            max_validate_acc.append(max(validate_acc))
            plot_train_validate_accuracy(train_acc, validate_acc)
            print(f"{i+1}", end=",")
        print()
        mean = statistics.mean(max_validate_acc)
        std = statistics.stdev(max_validate_acc, mean)
        print(f"mean validate R^2: {mean}")
        print(f"std validate R^2: {std}")
        results[f"{conv_filters},{kernel_size},{fc_node_count}"] = (mean, std)

    best_params, (best_mean, best_std) = max(results.items(), key=lambda item: item[1][0])
    print(f"{best_params}: {best_mean}, {best_std}")

    # with open("single_conv_2_fc_layers_3mers.json", "w") as f:
    #     json.dump(results, f)


if __name__ == "__main__":
    main()
