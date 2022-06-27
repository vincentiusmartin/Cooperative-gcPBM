import itertools
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, seqs, deltas, ids, *extra_features):
        self.sequences = seqs
        self.deltas = deltas
        self.extra_features = extra_features
        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].float()
        sequence = torch.transpose(sequence, 0, 1)

        extra_features = []
        for feature in self.extra_features:
            val = torch.tensor(feature[idx]).float()

            if val.dim() == 0:
                val = val.unsqueeze(0)

            # print(f"shape: {val.dim()}")
            extra_features.append(val)
            # print(f"seq_shape: {sequence.dim()}")
        deltas = torch.tensor(self.deltas[idx]).float()

        return sequence, deltas, *extra_features


def get_cross_validate_datasets(experiment_name, data_path, random_state, extra_features, mers=1,
                                return_dft=False):
    experiment_dict = {
        "ets1_ets1":
            {
                "labeled_data_path": os.path.join(data_path, "lbled_o1_selected.csv"),
                "training_data_path": os.path.join(data_path, "train_ets1_ets1.tsv"),
            },
        "ets1_runx1":
            {
                "labeled_data_path": os.path.join(data_path, "both_ori_plt_ets1_runx1.csv"),
                "training_data_path": os.path.join(data_path, "train_ets1_runx1.tsv"),
            },
    }

    experiment = experiment_dict[experiment_name]

    df_delta = pd.read_csv(experiment["labeled_data_path"])
    dft = pd.read_csv(experiment["training_data_path"], sep="\t")

    df_delta["delta"] = df_delta["two_median"] - df_delta["indiv_median"]

    dft = dft.merge(df_delta, on="Name")

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

    def make_categorical(element):
        return kmer_encoding_dict[element]

    vectorized_make_categorical = np.vectorize(make_categorical)

    sequences = np.array(seqs)  # convert to np array
    sequences = vectorized_make_categorical(sequences)  # encode nucleotides as integers
    sequences = torch.from_numpy(sequences)  # convert to torch tensor
    sequences = functional.one_hot(sequences)  # convert to one-hot encoding

    # Obtain target vector
    deltas = np.array(dft["delta"])
    deltas = torch.from_numpy(deltas)
    deltas = np.expand_dims(deltas, axis=1)

    extra_features_data = []

    if extra_features:
        if experiment_name == "ets1_ets1":
            if "site1_score" in extra_features:
                site1_scores = np.array(dft["site_wk_score"])
                extra_features_data.append(site1_scores)

            if "site2_score" in extra_features:
                site2_scores = np.array(dft["site_str_score"])
                extra_features_data.append(site2_scores)
        else:
            if "site1_score" in extra_features:
                site1_scores = np.array(dft["ets1_score"])
                extra_features_data.append(site1_scores)
            if "site2_score" in extra_features:
                site2_scores = np.array(dft["runx1_score"])
                extra_features_data.append(site2_scores)
            # runx1_scores = torch.from_numpy(runx1_scores)

        if "orientation" in extra_features:
            ori_encoding = {"+/+": 0, "+/-": 1, "-/+": 2, "-/-": 3}

            def encode_orientation(val):
                return ori_encoding[val]

            ori = np.array(dft["orientation"])
            vec_make_categorical = np.vectorize(encode_orientation)
            ori = vec_make_categorical(ori)
            ori = functional.one_hot(torch.from_numpy(ori), num_classes=4)
            extra_features_data.append(ori)

    k_fold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cross_validation_split_ids = k_fold.split(sequences)

    test_train_sets = []

    for train_ids, validate_ids in cross_validation_split_ids:
        scaler = StandardScaler().fit(deltas[train_ids])
        deltas_temp = scaler.transform(deltas)
        deltas_temp = np.squeeze(deltas_temp, axis=1)

        extra_features_train_sets = ([feature[train_ids] for feature in extra_features_data]
                                     if extra_features is not None else ())
        extra_features_validate_sets = ([feature[validate_ids] for feature in extra_features_data]
                                        if extra_features is not None else ())
        test_train_sets.append((
            SequenceDataset(sequences[train_ids], deltas_temp[train_ids], train_ids,
                            *extra_features_train_sets),
            SequenceDataset(sequences[validate_ids], deltas_temp[validate_ids], validate_ids,
                            *extra_features_validate_sets)
        ))

        if return_dft:
            test_train_sets[-1] = (*test_train_sets[-1], dft)

    # standardization needs to happen here. Also need a mechanism to turn it on and off.
    return test_train_sets
