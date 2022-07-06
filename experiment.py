import argparse
import json
import os
import statistics

import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import skorch
import torch

from datasets import get_datasets
from networks import NLayerCNN, SkorchNeuralNetRegressor


device = "cuda" if torch.cuda.is_available() else "cpu"

param_grid = {
    "regressor__module__conv_filters": [256],
    "regressor__module__fc_node_count": [512],
}


def plot_train_validate_accuracy(train_series, validate_series, file_name=None):
    epochs = len(train_series)
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(16, 6)

    ax1.plot(range(epochs), train_series, label="training")
    ax1.plot(range(epochs), validate_series, label="validation")

    ax1.axhline(y=0, color="grey", linestyle="--")
    ax1.axhline(y=max(validate_series), color="grey", linestyle="--")
    plt.ylim(0, max(validate_series))
    plt.xlabel("epochs")
    plt.ylabel("accuracy ($R^2$)")
    ax1.legend()
    plt.show()
    if file_name:
        plt.savefig(file_name, format="pdf")


def process_experiment_architecture_model(
        job_id,
        output_path,
        data_config,
        experiment_name,
        num_layers,
        mers,
        batch_size,
        kernel_sizes,
        include_affinities=False,
        patience=50,
        max_epochs=150,
        debug=False,
):
    datasets = get_datasets(experiment_name, data_config=data_config,
                            include_affinities=include_affinities, mers=mers)

    random_state = 1239283591

    datasets = shuffle(*datasets, random_state=random_state)
    Xs, y = datasets

    if debug:
        Xs, y = Xs[:25], y[:25]

    if any(True for param_comb in param_grid.values() if len(param_comb) > 1):
        net = SkorchNeuralNetRegressor(module=NLayerCNN,
                                       module__mers=mers,
                                       module__kernel_sizes=kernel_sizes,
                                       module__include_affinities=include_affinities,
                                       criterion=torch.nn.MSELoss,
                                       optimizer=torch.optim.Adam, device=device,
                                       max_epochs=max_epochs, batch_size=batch_size,
                                       callbacks=[("early_stopping",
                                                   skorch.callbacks.EarlyStopping(
                                                       patience=patience, monitor="valid_loss"))])

        regressor = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
        grid_search = GridSearchCV(regressor, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(Xs, y)

        filters_per_layer = grid_search.best_params_["regressor__module__conv_filters"]
        fc_node_count = grid_search.best_params_["regressor__module__fc_node_count"]
    else:
        filters_per_layer = param_grid["regressor__module__conv_filters"][0]
        fc_node_count = param_grid["regressor__module__fc_node_count"][0]

    file_path = os.path.join(output_path, f"task_{job_id}.json")

    with open(file_path, "w") as f:
        f.write(json.dumps([]))
        f.write("\n")

    # run five 5-fold cross-validations and take average to approximate R^2 on unseen data
    for i, random_state in enumerate(
            (3454832692, 3917820095, 851603617, 432544541, 4162995973)):
        # X must be: {sequences: [], site1_score: [], site2_score: []}
        Xs, y = shuffle(Xs, y, random_state=random_state)

        net = SkorchNeuralNetRegressor(module=NLayerCNN,
                                       module__conv_filters=filters_per_layer,
                                       module__fc_node_count=fc_node_count, module__mers=mers,
                                       module__kernel_sizes=kernel_sizes,
                                       module__include_affinities=include_affinities,
                                       criterion=torch.nn.MSELoss, optimizer=torch.optim.Adam,
                                       device=device, max_epochs=max_epochs, batch_size=batch_size,
                                       callbacks=[
                                           ("early_stopping",
                                            skorch.callbacks.EarlyStopping(patience=patience,
                                                                           monitor="valid_loss")),
                                           ("epoch_scoring",
                                            skorch.callbacks.EpochScoring(
                                                scoring="r2", lower_is_better=False))
                                       ])

        regressor = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())

        results = cross_validate(regressor, Xs, y, cv=5, scoring="r2", n_jobs=-1,
                                 return_estimator=True)

        histories = [estimator.regressor_.history for estimator in results["estimator"]]

        cv_max_validate_acc = [max(history[:, "r2"]) for history in histories]

        plot_train_validate_accuracy(histories[0][:, "train_loss"],
                                     histories[0][:, "valid_loss"],
                                     os.path.join(output_path, f"task_{job_id}.pdf"))
        # torch.save(net.state_dict(), os.path.join(output_path, f"task_{job_id}.pt"))

        mean = statistics.mean(cv_max_validate_acc)
        new_result = {
            "random_state": i,
            "num_layers": num_layers,
            "experiment": experiment_name,
            "mers": mers,
            "batch_size": batch_size,
            # "epochs": statistics.mean(epoch_counts),
            "cross_validation_test_r2": cv_max_validate_acc,
            "cv_test_r2_mean": mean,
            "cv_test_r2_mean_std": statistics.stdev(cv_max_validate_acc, mean),
            "patience": patience,
            "include_affinities": include_affinities,
        }

        new_result.update({
            "kernel_sizes": kernel_sizes,
            "conv_filters": filters_per_layer,
            "fc_node_count": fc_node_count,
        })

        with open(file_path, "r+") as f:
            json_list = json.load(f)
            json_list.append(new_result)
            f.seek(0)
            f.write(json.dumps(json_list))
            f.write("\n")
            f.truncate()


def main():
    # argument format:
    # <job_id> <output_path> <data_path> <"ets1_ets1"|"ets1_runx1"> <num_layers>
    # <mers> <batch_size> <layer_1_kernel_size>,...,<layer_n_kernel_size>
    # <extra_feature_1>,...,<extra_feature_n>

    parser = argparse.ArgumentParser(description="Train and cross validate models.")

    parser.add_argument("job_id", type=int)
    parser.add_argument("output_path", type=str)
    parser.add_argument("data_config", type=str)
    parser.add_argument("experiment", type=str)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("mers", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("kernel_sizes", type=str, help="this should be comma-separated (no spaces)")
    parser.add_argument("include_affinities", type=str)

    args = parser.parse_args()

    job_id = args.job_id
    output_path = args.output_path
    data_config = args.data_config
    experiment = args.experiment
    num_layers = args.num_layers
    mers = args.mers
    batch_size = args.batch_size
    kernel_sizes = [int(k) for k in args.kernel_sizes.split(",")]
    include_affinities = bool(args.include_affinities.lower() in ("TRUE", "YES", "T", "Y"))

    print(f"experiment: {experiment}, layers: {num_layers}, job_id:{job_id}")
    process_experiment_architecture_model(job_id, output_path, data_config, experiment, num_layers,
                                          mers, batch_size, kernel_sizes, include_affinities)


if __name__ == "__main__":
    main()

    # process_experiment_architecture_model(1,
    #                                       "/Users/kylepinheiro/compsci260/dl_cooperativity/output",
    #                                       "/Users/kylepinheiro/research_code/data", "ets1_runx1", 2,
    #                                       1, 32, [8, 8], True, patience=20, max_epochs=50,
    #                                       debug=False)
