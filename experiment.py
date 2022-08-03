import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import skorch
import torch

from datasets import get_datasets
from networks import SkorchNeuralNetRegressor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def plot_loss_curves(train_series, validate_series, file_name=None):
    epochs = len(train_series)
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(16, 6)

    ax1.plot(range(epochs), train_series, label="training")
    ax1.plot(range(epochs), validate_series, label="validation")

    ax1.axhline(y=0, color="grey", linestyle="--")
    ax1.axhline(y=min(validate_series), color="grey", linestyle="--")
    plt.ylim(0, max(validate_series))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    ax1.legend()
    plt.show()
    if file_name:
        plt.savefig(file_name, format="pdf")


class CustomPrefixCheckpoint(skorch.callbacks.Checkpoint):
    def initialize(self):
        self.fn_prefix = str(id(self))
        return super(CustomPrefixCheckpoint, self).initialize()


def process_experiment(
        job_id,
        output_path,
        data_config,
        experiment_name,
        num_layers,
        mers,
        batch_size,
        kernel_widths,
        include_affinities=False,
        pool=False,
        dropout_rate=0,
        weight_decay=0,
        lr=0.001,
        conv_filters=256,
        fc_node_count=512,
        patience=50,
        max_epochs=200,
        debug=False,
):
    net_params = {
        "device": device,
        "optimizer": torch.optim.Adam,
        "optimizer__lr": lr,
        "optimizer__weight_decay": weight_decay,
        "iterator_train__shuffle": True,
        "criterion": torch.nn.MSELoss,
        "module__mers": mers,
        "module__kernel_widths": kernel_widths,
        "module__include_affinities": include_affinities,
        "module__pool": pool,
        "module__conv_filters": conv_filters,
        "module__fc_node_count": fc_node_count,
        "max_epochs": max_epochs,
        "module__dropout": dropout_rate,
        "batch_size": batch_size,
        "verbose": 0,
    }
    datasets = get_datasets(experiment_name, data_config=data_config,
                            include_affinities=include_affinities, mers=mers)
    temp_dir = f"./output/{job_id}_temp"
    random_state = 1239283591

    datasets = shuffle(*datasets, random_state=random_state)
    Xs, y = datasets

    if debug:
        Xs, y = Xs[:50], y[:50]

    file_path = os.path.join(output_path, f"task_{job_id}.json")

    with open(file_path, "w") as f:
        f.write(json.dumps([]))
        f.write("\n")

    # run five 5-fold cross-validations and take average to approximate R^2 on unseen data
    for i, random_state in enumerate(
            (3454832692, 3917820095, 851603617, 432544541, 4162995973)):
        # Xs must be: {sequences: [], site1_score: [], site2_score: []}
        Xs, y = shuffle(Xs, y, random_state=random_state)

        net = SkorchNeuralNetRegressor(
            callbacks=[
                skorch.callbacks.EarlyStopping(patience=patience, monitor="valid_loss"),
                CustomPrefixCheckpoint(monitor="valid_loss_best", load_best=True, dirname=temp_dir),
            ],
            **net_params,
        )

        regressor = TransformedTargetRegressor(regressor=net, transformer=StandardScaler(),
                                               check_inverse=False)

        results = cross_validate(regressor, Xs, y, cv=5, scoring="r2", n_jobs=-1,
                                 return_estimator=True, return_train_score=True)

        histories = [estimator.regressor_.history for estimator in results["estimator"]]

        if i == 1:
            plot_loss_curves(histories[0][:, "train_loss"], histories[0][:, "valid_loss"],
                             os.path.join(output_path, f"task_{job_id}.pdf"))

        epochs = [min(enumerate(history[:, "valid_loss"], start=0), key=lambda x: x[1])[0]
                  for history in histories]

        shutil.rmtree(temp_dir)

        new_result = {
            "random_state": i,
            "num_layers": num_layers,
            "experiment": experiment_name,
            "mers": mers,
            "batch_size": batch_size,
            "epochs": epochs,
            "cross_validation_test_r2": list(results["test_score"]),
            "cross_validation_train_r2": list(results["train_score"]),
            "cv_test_r2_mean": results["test_score"].mean(),
            "cv_test_r2_mean_std": results["test_score"].std(),
            "patience": patience,
            "max_epochs": max_epochs,
            "kernel_widths": kernel_widths,
            "conv_filters": net_params["module__conv_filters"],
            "fc_node_count": net_params["module__fc_node_count"],
            "include_affinities": include_affinities,
            "max_pooling": pool,
            "dropout": dropout_rate,
            "decay": weight_decay,
            "lr": lr,
        }

        with open(file_path, "r+") as f:
            json_list = json.load(f)
            json_list.append(new_result)
            f.seek(0)
            f.write(json.dumps(json_list))
            f.write("\n")
            f.truncate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and cross validate models.")

    parser.add_argument("job_id", type=int)
    parser.add_argument("output_path", type=str)
    parser.add_argument("data_config", type=str)
    parser.add_argument("experiment", type=str)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("mers", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("kernel_widths", type=str, help="must be comma-separated with no spaces")
    parser.add_argument("include_affinities", type=str)
    parser.add_argument("pool", type=str)
    parser.add_argument("dropout", type=float)
    parser.add_argument("decay", type=float)
    parser.add_argument("lr", type=float)
    parser.add_argument("conv_filters", type=int)
    parser.add_argument("fc_node_count", type=int)
    args = parser.parse_args()

    kernel_widths = [int(k) for k in args.kernel_widths.split(",")]
    include_affinities = bool(args.include_affinities.upper() in ("TRUE", "YES", "T", "Y"))
    pool = bool(args.pool.upper() in ("TRUE", "YES", "T", "Y"))

    print(f"experiment: {args.experiment}, layers: {args.num_layers}, job_id:{args.job_id}")
    process_experiment(
        args.job_id,
        args.output_path,
        args.data_config,
        args.experiment,
        args.num_layers,
        args.mers,
        args.batch_size,
        kernel_widths,
        include_affinities,
        pool,
        args.dropout,
        args.decay,
        args.lr,
        args.conv_filters,
        args.fc_node_count,
    )
