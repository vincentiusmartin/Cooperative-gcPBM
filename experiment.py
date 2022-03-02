import itertools
import json
import os
import statistics
import sys

import matplotlib.pyplot as plt
from sklearn import metrics
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from datasets import get_cross_validate_datasets
from networks import CNN, TwoLayerCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

architecture_maps = {
    "one_layer_cnn": {
        "model": CNN,
        "params": ("kernel_size",),
        "grid": {
            "conv_filters": [16, 32, 64, 128, 256],
            "fc_layer_nodes": [128, 256, 512],
        },
    },
    "two_layer_cnn": {
        "model": TwoLayerCNN,
        "params": ("kernel_size", "kernel2_size"),
        "grid": {
            "conv_filters": [16, 32, 64, 128, 256],
            "conv2_filters": [16, 32, 64, 128, 256],
            "fc_layer_nodes": [128, 256, 512],
        },
    },
}


def train(optimizer, loss_fn, model, dataloader):
    model.train()

    for X, y_true in dataloader:
        X = X.to(device)
        y_true = y_true.to(device)
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
            X = X.to(device)
            y = y.to("cpu")
            y_hat = model(X).to("cpu")
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
            X = X.to(device)
            y = y.to("cpu")
            y_hat = model(X).to("cpu")
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


def process_experiment_architecture_model(job_id, output_path, data_path, experiment_name,
                                          architecture_name, mers, *params):
    architecture = architecture_maps[architecture_name]

    # validate the params and move to a dictionary or variables
    if len(architecture["params"]) != len(params):
        raise AttributeError(f"architecture: {architecture_name} requires "
                             f"{len(architecture['params'])} parameters:"
                             f"\n {architecture['params']}")

    params = dict(zip(architecture["params"], (int(param) for param in params)))

    NeuralNetwork = architecture["model"]
    grid_ranges = architecture["grid"]

    # give me all possible combinations of the grid ranges
    grid = list(itertools.product(*grid_ranges.values()))

    batch_size = 64
    epochs = 30

    random_state = 1239283591
    cross_validation_splits = get_cross_validate_datasets(experiment_name, data_path=data_path,
                                                          random_state=random_state,
                                                          mers=mers)

    cv_means = []
    for grid_params in grid:
        # Maybe use skorch for some part of this?

        cv_max_validate_acc = []
        for j, (train_data, validate_data) in enumerate(cross_validation_splits):
            torch.manual_seed(random_state)
            print(f"cross-validation: {j+1}")

            print(grid_params)
            print(params)
            net = NeuralNetwork(*grid_params, mers=mers, **params).to(device)
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
            validate_dataloader = DataLoader(validate_data, batch_size=batch_size)

            optimizer = torch.optim.Adam(net.parameters())
            loss_fn = MSELoss()

            train_acc = []
            validate_acc = []

            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Training the model 1 time                           #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            for t in range(epochs):
                train(optimizer, loss_fn, net, train_dataloader)
                train_acc.append(get_r2(net, train_dataloader))
                validate_acc.append(get_r2(net, validate_dataloader))

            cv_max_validate_acc.append(max(validate_acc))
            # plot_train_validate_accuracy(train_acc, validate_acc)

        cv_means.append((grid_params, statistics.mean(cv_max_validate_acc)))

    best_grid_params = max(cv_means, key=lambda x: x[1])[0]

    print(best_grid_params)

    file_path = os.path.join(output_path, f"task_{job_id}.json")

    for i, random_state in enumerate(
            (3454832692, 3917820095, 851603617, 432544541, 4162995973)):
        cross_validation_splits = get_cross_validate_datasets(experiment_name, data_path=data_path,
                                                              random_state=random_state,
                                                              mers=mers)
        with open(file_path, "w") as f:
            f.write(json.dumps([]))

        cv_max_validate_acc = []
        for j, (train_data, validate_data) in enumerate(cross_validation_splits):
            torch.manual_seed(random_state)
            print(f"cross-validation: {j+1}")

            net = NeuralNetwork(*best_grid_params, mers=mers, **params).to(device)
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
            validate_dataloader = DataLoader(validate_data, batch_size=batch_size)

            optimizer = torch.optim.Adam(net.parameters())
            loss_fn = MSELoss()

            train_acc = []
            validate_acc = []

            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Training the model 1 time                           #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            for t in range(epochs):
                train(optimizer, loss_fn, net, train_dataloader)
                train_acc.append(get_r2(net, train_dataloader))
                validate_acc.append(get_r2(net, validate_dataloader))

            cv_max_validate_acc.append(max(validate_acc))
            # plot_train_validate_accuracy(train_acc, validate_acc)

        mean = statistics.mean(cv_max_validate_acc)
        new_result = {
            "architecture": architecture_name,
            "experiment": experiment_name,
            "mers": mers,
            "cross_validation_test_r2": cv_max_validate_acc,
            "cv_test_r2_mean": mean,
            "cv_test_r2_mean_std": statistics.stdev(cv_max_validate_acc, mean)
        }

        new_result.update(params)

        new_result.update(dict(zip(list(grid_ranges.keys()), best_grid_params)))

        # best_params, (best_mean, best_std) = max(results.items(), key=lambda item: item[1][0])
        # print(f"{best_params}: {best_mean}, {best_std}")

        with open(file_path, "r+") as f:
            json_list = json.load(f)

            json_list.append(new_result)
            f.seek(0)
            f.write(json.dumps(json_list))
            f.truncate()


if __name__ == "__main__":
    # process_experiment_architecture_model(experiment, architecture, "/null")
    # argument format:
    # <job_id> <output_path> <data_path> <ets1_ets1|ets1_runx1> <architecture>
    # <mers> <kernel_size> <num_conv_filters>
    print(sys.argv)
    job_id = sys.argv[1]
    output_path = sys.argv[2]
    data_path = sys.argv[3]  # "/Users/kylepinheiro/research_code"
    experiment = sys.argv[4]  # "ets1_ets1"
    architecture = sys.argv[5]  # "one_layer_cnn"
    mers = int(sys.argv[6])
    arch_params = sys.argv[7:]

    print(f"experiment: {experiment}, architecture: {architecture}, job_id:{job_id}")
    process_experiment_architecture_model(job_id, output_path, data_path, experiment, architecture,
                                          mers, *arch_params)
