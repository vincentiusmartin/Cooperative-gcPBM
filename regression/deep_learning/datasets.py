import itertools
import json
import os

import numpy as np
import pandas as pd
from skorch.helper import SliceDict
import torch
from torch.nn import functional


def get_dataframe(config_file, experiment_name):
    with open(config_file, "r") as f:
        exp_dict = json.load(f)

    experiment = exp_dict[experiment_name]

    # potentially consolidate these pairs of data files.
    df_delta = pd.read_csv(experiment["labeled_data_path"])
    dft = pd.read_csv(experiment["training_data_path"], sep="\t")

    if experiment_name == "ets1_ets1":
        df_delta["delta"] = df_delta["two_median"] - df_delta["indiv_median"]
    else:
        df_delta["delta"] = df_delta["intensity_y"] - df_delta["intensity_x"]

    return dft.merge(df_delta, on="Name")


def get_datasets(experiment_name, data_config, include_affinities=True, mers=2):
    """Retrieve input features and target labels.

    Args:
        experiment_name: "ets1_ets1" or "ets1_runx1".
        data_config: file path to json.
        include_affinities: boolean to specify whether to include affinity as input feature.
        mers: number of mers.

    Returns:
        X: SliceDict containing input features
        y: pytorch.tensor containing target values
    """
    dft = get_dataframe(data_config, experiment_name)

    # for ets1_ets1 experiments, only use data that was classified as cooperative
    if experiment_name == "ets1_ets1":
        dft = dft[dft["label_x"] == "cooperative"]
        dft = dft.reset_index()

    # Generate input sequence data depending on k-mer length
    if mers == 1:
        seqs = dft["Sequence"].str.split(pat="", expand=True)
        seqs = seqs.loc[:, 1:36]
    elif mers > 1:
        seqs = pd.DataFrame()

        for i in range(1, 36-mers+2):
            seqs[str(i)] = dft["Sequence"].str.slice(i-1, i+mers-1)
    else:
        raise ValueError("mers must be set to a positive integer")

    kmer_encoding_dict = {kmer: index for index, kmer in
                          enumerate("".join(tup) for tup in
                                    itertools.product(["A", "T", "C", "G"], repeat=mers))
                          }

    sequences = np.array(seqs)  # convert to np array
    vectorized_make_categorical = np.vectorize(lambda element: kmer_encoding_dict[element])
    sequences = vectorized_make_categorical(sequences)  # encode nucleotides as integers
    sequences = torch.from_numpy(sequences)  # convert to torch tensor
    sequences = functional.one_hot(sequences)  # convert to one-hot encoding
    sequences = sequences.float()
    X = {"sequences": sequences}

    # Obtain target vector
    y = np.array(dft["delta"])
    y = torch.from_numpy(y).reshape(-1, 1).float()

    if include_affinities:
        if experiment_name == "ets1_ets1":
            X["site1_scores"] = np.array(dft["site_wk_score"])
            X["site2_scores"] = np.array(dft["site_str_score"])
        else:
            X["site1_scores"] = np.array(dft["ets1_score"])
            X["site2_scores"] = np.array(dft["runx1_score"])

        X["site1_scores"] = torch.from_numpy(X["site1_scores"]).reshape(-1, 1).float()
        X["site2_scores"] = torch.from_numpy(X["site2_scores"]).reshape(-1, 1).float()

    return SliceDict(**X), y

