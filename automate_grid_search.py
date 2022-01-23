#!/usr/bin/env python
# coding: utf-8

# this code was originally authored by: Vincentius Martin: vincentius.martin@duke.edu
# modified substantially by: Kyle Pinheiro: kyle.pinheiro@duke.edu
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils import shuffle

from chip2probe.modeler.cooptrain import CoopTrain

# define hyper-parameter grid
models = {
    "random_forest_regression": {
        "class": RandomForestRegressor,
        "param_grid": {
            "n_estimators": [32, 64, 128, 256],
            "max_features": ["sqrt", None],
            "max_depth": [2, 4, 8, 16, 32, 64, 128],
            "min_samples_leaf": [2, 4, 8],
            "min_samples_split": [2, 4, 8],
            "min_impurity_decrease": [0, .2, .4, .6, .8, 1],
        }
    },
    "support_vector_regression": {
        "class": SVR,
        "param_grid": [
            {
                "C": [.25, .5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                      32768, 65536],
                "gamma": [.000001, .00001, .0001, .001, .01, .1, 1, 2, 10, 100],
                "kernel": ["rbf"],
                "epsilon": [.125, .25, .5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            },
        ],
    },
}

# define feature set grid
experiment_dict = {
    "ets1_ets1":
    {
        "labeled_data_path": "./data/lbled_o1_selected.csv",
        "training_data_path": "./data/train_ets1_ets1.tsv",
        "position_keys": ("site_str_pos", "site_wk_pos"),
        "feature_dict": {
            "distance": {"type": "numerical"},
            "affinity": {"colnames": ("site_str_score", "site_wk_score")},
            "orientation": {"relative": True, "one_hot": True,
                            "pos_cols": {"site_str_pos": "site_str_ori",
                                         "site_wk_pos": "site_wk_ori"}},
            "shape_in": {"seqin": 5, "poscols": ["site_str_pos", "site_wk_pos"],
                         "smode": "relative"},
            "shape_out": {"seqin": -2, "poscols": ["site_str_pos", "site_wk_pos"],
                          "smode": "relative"},
            "sequence_in": {"seqin": 5, "poscols": ["site_str_pos", "site_wk_pos"],
                            "namecol": "Name", "smode": "relative"},
            "sequence_out": {"seqin": -3, "poscols": ["site_str_pos", "site_wk_pos"],
                             "namecol": "Name", "smode": "relative"},
            "position": {"colnames": ("site_str_pos", "site_wk_pos")},
        },
    },
    "ets1_runx1":
    {
        "labeled_data_path": "./data/both_ori_plt_ets1_runx1.csv",
        "training_data_path": "./data/train_ets1_runx1.tsv",
        "position_keys": ("ets1_pos", "runx1_pos"),
        "feature_dict": {
            "distance": {"type": "numerical"},
            "affinity": {"colnames": ("ets1_score", "runx1_score")},
            "orientation": {"relative": True, "one_hot": True,
                            "pos_cols": {"ets1_pos": "ets1_ori", "runx1_pos": "runx1_ori"}},
            "shape_in": {"seqin": 5, "poscols": ["ets1_pos", "runx1_pos"], "smode": "positional"},
            "shape_out": {"seqin": -2, "poscols": ["ets1_pos", "runx1_pos"], "smode": "positional"},
            "sequence_in": {"seqin": 5, "poscols": ["ets1_pos", "runx1_pos"], "namecol": "Name",
                            "smode": "relative"},
            "sequence_out": {"seqin": -3, "poscols": ["ets1_pos", "runx1_pos"],
                             "namecol": "Name", "smode": "relative"},
            "position": {"colnames": ("ets1_pos", "runx1_pos")},
        },
    },
}


def get_features(df, keys, exp):
    ct = CoopTrain(df)

    feature_dict = {key: exp["feature_dict"][key] for key in keys
                    if key in exp["feature_dict"].keys()}

    X = ct.get_feature_all(feature_dict)
    X = X.values.tolist()
    X = StandardScaler().fit_transform(X)
    ytrue = np.array(ct.df["delta"].values.tolist())

    return X, ytrue


def process_experiment_feature_set_model(experiment_name, feature_set, model, file_path):
    experiment = experiment_dict[experiment_name]
    dfdelta = pd.read_csv(experiment["labeled_data_path"])
    dft = pd.read_csv(experiment["training_data_path"], sep="\t")

    dfdelta["delta"] = dfdelta["two_median"] - dfdelta["indiv_median"]
    dfdelta = dfdelta[["Name", "delta"]]

    dft = dft.merge(dfdelta, on="Name")

    # These two lines enable/disable filtering on "cooperative" classification
    #  dft = dft[dft["label"] == "cooperative"]
    #  dft = dft.reset_index()

    model_grid = models[model]
    # print(f"model: {model}, experiment: {experiment_name} feature_set: {feature_set}")

    X, ytrue = get_features(dft, feature_set, experiment)
    X, ytrue = shuffle(X, ytrue, random_state=1239283591)

    param = {}

    if model_grid["class"].__name__ == "RandomForestRegressor":
        param["random_state"] = 1239283591

    grid_search = GridSearchCV(model_grid["class"](**param), model_grid["param_grid"], cv=5,
                               refit=True, n_jobs=-1, verbose=2)
    grid_search.fit(X, ytrue)

    params = grid_search.best_params_

    with open(file_path, "w") as f:
        f.write(json.dumps([]))

    # do 5 cross-validations so we can aggregate data across more than one replication of
    # cross-validation (5 randomly-generated integers for random seeds)
    for rand_state in (3454832692, 3917820095, 851603617, 432544541, 4162995973):
        X, ytrue = shuffle(X, ytrue, random_state=rand_state)

        if model_grid["class"].__name__ == "RandomForestRegressor":
            params["random_state"] = rand_state

        cv_results = cross_validate(model_grid["class"](**params), X, ytrue, cv=5, scoring="r2")

        new_result = {
            "model": model,
            "experiment": experiment_name,
            "feature_set": feature_set,
            "params": params,
            "param_search_test_r2s": [
                grid_search.cv_results_[f"split{x}_test_score"][grid_search.best_index_]
                for x in range(0, 5)
            ],
            "cross_validation_test_r2": list(cv_results["test_score"]),
            "cv_test_r2_mean": cv_results["test_score"].mean(),
        }

        with open(file_path, "r+") as f:
            json_list = json.load(f)

            json_list.append(new_result)
            f.seek(0)
            f.write(json.dumps(json_list))
            f.truncate()


if __name__ == "__main__":
    # argument format:
    # <job_id> <path> <ets1_ets1|ets1_runx1> <random_forest_regression|support_vector_regression>
    # <feature_1>,<feature_2>,...,<feature_n>
    print(sys.argv)
    job_id = sys.argv[1]
    path = sys.argv[2]
    experiment = sys.argv[3]
    model = sys.argv[4]
    feature_set = sys.argv[5].split(",")

    # results will be placed in file with unique name by job id
    file_path = os.path.join(path, f"task_{job_id}.json")

    print(f"experiment: {experiment}, model: {model}, feature_set: {feature_set}, job_id:{job_id}")
    process_experiment_feature_set_model(experiment, feature_set, model, file_path)
