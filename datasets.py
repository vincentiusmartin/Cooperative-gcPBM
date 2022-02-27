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
    def __init__(self, seqs, deltas, ids):
        self.sequences = seqs[ids]
        self.deltas = deltas[ids]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].float()
        sequence = torch.transpose(sequence, 0, 1)

        return sequence, self.deltas[idx].float()


def get_cross_validate_datasets(experiment_name, data_path, random_state, mers=1):
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

        for i in range(1, 36-mers+1):
            seqs[str(i)] = dft["Sequence"].str.slice(i, i+mers)
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
    deltas = dft["delta"]
    deltas = np.expand_dims(deltas, axis=1)

    # scaling needs to be moved to where it can be done on trained data
    scaler = StandardScaler().fit(deltas)  # need to only apply this to the input
    deltas = scaler.transform(deltas)
    deltas = np.squeeze(deltas, axis=1)
    deltas = torch.from_numpy(deltas)

    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
    k_fold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cross_validation_split_ids = k_fold.split(sequences)

    return [
        (SequenceDataset(sequences, deltas, train_ids),
         SequenceDataset(sequences, deltas, validate_ids))
        for train_ids, validate_ids in cross_validation_split_ids]
