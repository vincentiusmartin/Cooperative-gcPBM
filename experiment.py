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
from networks import MultiInputCNN, NLayerCNN


device = "cuda" if torch.cuda.is_available() else "cpu"

architecture_maps = {
    "multi_input_one_layer_cnn": {
        "model": MultiInputCNN,
        "params": ("kernel_size",),
        "grid": {
            "conv_filters": [16, 32, 64, 128, 256],
            "fc_layer_nodes": [128, 256, 512],
        },
    },
    "four_layer_cnn": {
        "model": NLayerCNN,
        "params": ("kernel_size", 4),
        "grid": {
            "conv_filters": [[256],  # layer 1
                             [256],  # layer 2
                             [256],  # layer 3
                             [256],
                             ],
            "fc_layer_nodes": [256, 512],
        },
    },
    "three_layer_cnn": {
        "model": NLayerCNN,
        "params": ("kernel_size", 3),
        "grid": {
            "conv_filters": [[256],  # layer 1
                             [256],  # layer 2
                             [256],  # layer 3
                             ],
            "fc_layer_nodes": [512],
        },
    },
    "two_layer_cnn": {
        "model": NLayerCNN,
        "params": ("kernel_size", 2),
        "grid": {
            "conv_filters": [[256],  # layer 1
                             [256],  # layer 2
                             ],
            "fc_layer_nodes": [512],
        },
    },
    "one_layer_cnn": {
        "model": NLayerCNN,
        "params": ("kernel_size", 1),
        "grid": {
            "conv_filters": [[256],  # layer 1
                             ],
            "fc_layer_nodes": [512],
        },
    },
}


def train(optimizer, loss_fn, model, dataloader):
    model.train()
    # for i in range(0, len(data_point)):
    #     data_point[i] = data_point[i].to(device)
    #
    # X, y_true, *extra_features = data_point

    for X, y_true, *extra_features in dataloader:
        for i in range(0, len(extra_features)):
            extra_features[i] = extra_features[i].to(device)

        X = X.to(device)
        y_true = y_true.to(device)
        y_hat = model(X, *extra_features)
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
        for X, y, *extra_features in dataloader:
            for i in range(0, len(extra_features)):
                extra_features[i] = extra_features[i].to(device)

            X = X.to(device)
            y = y.to("cpu")
            y_hat = model(X, *extra_features).to("cpu")
            y_hat = y_hat.flatten()
            y_hat_arr += list(y_hat)
            y_true_arr += list(y)

    return metrics.r2_score(y_true_arr, y_hat_arr)


def plot_train_validate_accuracy(train_series, validate_series, file_name=None):
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
    if file_name:
        plt.savefig(file_name, format="pdf")


def process_experiment_architecture_model(job_id, output_path, data_path, experiment_name,
                                          architecture_name, mers, batch_size, kernel_sizes,
                                          extra_features):

    extra_feature_count = 0

    if extra_features is not None:
        extra_feature_count = len(extra_features)

        if "orientation" in extra_features:
            extra_feature_count += 3

    architecture = architecture_maps[architecture_name]

    # validate the params and move to a dictionary or variables
    # if len(architecture["params"]) != len(params):
    #     raise AttributeError(f"architecture: {architecture_name} requires "
    #                          f"{len(architecture['params'])} parameters:"
    #                          f"\n {architecture['params']}")

    _, num_kernels = architecture["params"]

    NeuralNetwork = architecture["model"]
    grid_ranges = architecture["grid"]

    # give me all possible combinations of the grid ranges

    grid = list(itertools.product(itertools.product(*grid_ranges["conv_filters"]),
                                  grid_ranges["fc_layer_nodes"]))

    random_state = 1239283591
    cross_validation_splits = get_cross_validate_datasets(experiment_name, data_path=data_path,
                                                          random_state=random_state,
                                                          extra_features=extra_features, mers=mers)

    max_epochs = 120
    patience = 20

    cv_means = []
    for grid_params in grid:
        # Maybe use skorch for some part of this?

        cv_max_validate_acc = []
        for j, (train_data, validate_data) in enumerate(cross_validation_splits):
            # torch.manual_seed(random_state)

            net = NeuralNetwork(*grid_params, mers=mers, kernel_sizes=kernel_sizes,
                                extra_feature_count=extra_feature_count).to(device)
            # print(f"model device: {next(net.parameters()).device}")
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
            validate_dataloader = DataLoader(validate_data, batch_size=batch_size)

            optimizer = torch.optim.Adam(net.parameters())
            loss_fn = MSELoss()

            # train_acc = []
            validate_acc = []

            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Training the model 1 time                           #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            best, best_epoch = (-1., 0)
            for t in range(max_epochs):
                train(optimizer, loss_fn, net, train_dataloader)
                # train_acc.append(get_r2(net, train_dataloader))
                validate_r2 = get_r2(net, validate_dataloader)
                validate_acc.append(validate_r2)

                if validate_r2 > best:
                    best, best_epoch = validate_r2, t
                elif t > patience + best_epoch:
                    break

            cv_max_validate_acc.append(max(validate_acc))
            # plot_train_validate_accuracy(train_acc, validate_acc)

        cv_means.append((grid_params, statistics.mean(cv_max_validate_acc)))

    best_grid_params = max(cv_means, key=lambda x: x[1])[0]

    file_path = os.path.join(output_path, f"task_{job_id}.json")

    with open(file_path, "w") as f:
        f.write(json.dumps([]))
        f.write("\n")

    for i, random_state in enumerate(
            (3454832692, 3917820095, 851603617, 432544541, 4162995973)):
        cross_validation_splits = get_cross_validate_datasets(experiment_name, data_path=data_path,
                                                              random_state=random_state,
                                                              extra_features=extra_features,
                                                              mers=mers)

        epoch_counts = []
        cv_max_validate_acc = []
        for j, (train_data, validate_data) in enumerate(cross_validation_splits):
            # torch.manual_seed(random_state)

            net = NeuralNetwork(*best_grid_params, mers=mers, kernel_sizes=kernel_sizes,
                                extra_feature_count=extra_feature_count).to(device)
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
            validate_dataloader = DataLoader(validate_data, batch_size=batch_size)

            optimizer = torch.optim.Adam(net.parameters())
            loss_fn = MSELoss()

            # train_acc = []
            validate_acc = []

            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Training the model 1 time                           #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            best, best_epoch = (-1., 0)
            for t in range(max_epochs):
                epoch_counts.append(t)
                train(optimizer, loss_fn, net, train_dataloader)
                # train_acc.append(get_r2(net, train_dataloader))
                validate_r2 = get_r2(net, validate_dataloader)
                validate_acc.append(validate_r2)

                if validate_r2 > best:
                    best, best_epoch = validate_r2, t
                elif t > patience + best_epoch:
                    break

            cv_max_validate_acc.append(max(validate_acc))
            # plot_train_validate_accuracy(train_acc, validate_acc,
            #                              os.path.join(output_path, f"task_{job_id}.pdf"))
            # torch.save(net.state_dict(), os.path.join(output_path, f"task_{job_id}.pt"))

        mean = statistics.mean(cv_max_validate_acc)
        new_result = {
            "random_state": i,
            "architecture": architecture_name,
            "experiment": experiment_name,
            "mers": mers,
            "batch_size": batch_size,
            "epochs": statistics.mean(epoch_counts),
            "cross_validation_test_r2": cv_max_validate_acc,
            "cv_test_r2_mean": mean,
            "cv_test_r2_mean_std": statistics.stdev(cv_max_validate_acc, mean),
            "patience": patience,
            "extra_features": extra_features,
        }

        new_result.update({"kernel_sizes": kernel_sizes})

        # need to update this too.. grid_ranges.keys() no longer has everything I want
        new_result.update(dict(zip(list(grid_ranges.keys()), best_grid_params)))

        # best_params, (best_mean, best_std) = max(results.items(), key=lambda item: item[1][0])
        # print(f"{best_params}: {best_mean}, {best_std}")
        with open(file_path, "r+") as f:
            json_list = json.load(f)
            json_list.append(new_result)
            f.seek(0)
            f.write(json.dumps(json_list))
            f.write("\n")
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
    batch_size = int(sys.argv[7])
    kernel_sizes = sys.argv[8].split(",")
    extra_features = sys.argv[9]
    if extra_features == "":
        extra_features = None
    else:
        extra_features = extra_features.split(",")

    kernel_sizes = [int(k) for k in kernel_sizes]

    print(f"experiment: {experiment}, architecture: {architecture}, job_id:{job_id}")
    process_experiment_architecture_model(job_id, output_path, data_path, experiment, architecture,
                                          mers, batch_size, kernel_sizes, extra_features)
