import json
import os

import matplotlib.pyplot as plt
from sklearn import metrics
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from datasets import get_cross_validate_datasets
from networks import NLayerCNN


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(optimizer, loss_fn, model, dataloader):
    model.train()
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

    max_epochs = 250
    patience = 250

    file_path = os.path.join(output_path, f"task_{job_id}.json")

    with open(file_path, "w") as f:
        f.write(json.dumps([]))
        f.write("\n")

    random_state = 3454832692
    cross_validation_splits = get_cross_validate_datasets(experiment_name, data_path=data_path,
                                                          random_state=random_state,
                                                          extra_features=extra_features,
                                                          mers=mers)
    j, (train_data, validate_data) = next(enumerate(cross_validation_splits))

    net = NLayerCNN(CONV_FILTERS, FC_LAYER_COUNT, mers=mers, kernel_sizes=kernel_sizes,
                    extra_feature_count=extra_feature_count, pool="max").to(device)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    validate_dataloader = DataLoader(validate_data, shuffle=True, batch_size=batch_size)

    optimizer = torch.optim.Adam(net.parameters())
    loss_fn = MSELoss()

    train_acc = []
    validate_acc = []

    # Training loop
    best, best_epoch = (-1., 0)
    for t in range(max_epochs):
        train(optimizer, loss_fn, net, train_dataloader)
        train_acc.append(get_r2(net, train_dataloader))
        validate_r2 = get_r2(net, validate_dataloader)
        validate_acc.append(validate_r2)

        if validate_r2 > best:
            best, best_epoch = validate_r2, t

            torch.save(net.state_dict(), os.path.join(output_path, f"task_{job_id}.pt"))
        elif t > patience + best_epoch:
            break

    plot_train_validate_accuracy(train_acc, validate_acc,
                                 os.path.join(output_path, f"task_{job_id}.pdf"))

    new_result = {
        "architecture": architecture_name,
        "experiment": experiment_name,
        "mers": mers,
        "batch_size": batch_size,
        "epochs": best_epoch,
        "cv_test": validate_r2,
        "best_cv_test": best,
        "patience": patience,
        "extra_features": extra_features,
    }

    new_result.update({"kernel_sizes": kernel_sizes})

    # need to update this too.. grid_ranges.keys() no longer has everything I want
    new_result.update(dict(zip(("conv_filters", "fc_layer_nodes"), (CONV_FILTERS, FC_LAYER_COUNT))))

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
    # manually adjust this dictionary to reflect the model to be trained
    g = {"num_layers": 4, "experiment": "ets1_runx1", "mers": 2, "batch_size": 32,
         "extra_features": ["site1_score", "site2_score"],
         "kernel_sizes": [4, 6, 3, 5], "conv_filters": [256, 256, 256, 256], "fc_layer_nodes": 512,
         "file_path": "four_layer_best",
         }

    job_id = g["file_path"]

    output_path = os.getcwd()  # or replace with path to output
    data_path = "<insert path here>"
    experiment = g["experiment"]
    num_layers = g["num_layers"]
    mers = g["mers"]
    batch_size = g["batch_size"]
    kernel_sizes = g["kernel_sizes"]
    extra_features = g["extra_features"]

    kernel_sizes = [int(k) for k in kernel_sizes]

    CONV_FILTERS = g["conv_filters"]
    FC_LAYER_COUNT = g["fc_layer_nodes"]

    print(f"experiment: {experiment}, num_layers: {num_layers}, job_id:{job_id}")
    process_experiment_architecture_model(job_id, output_path, data_path, experiment, num_layers,
                                          mers, batch_size, kernel_sizes, extra_features)
