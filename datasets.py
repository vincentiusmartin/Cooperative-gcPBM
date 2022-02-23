import itertools
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional
from torch.utils.data import Dataset

os.chdir("/Users/kylepinheiro/research_code")


class SequenceDataset(Dataset):
    def __init__(self, seqs, deltas, train=False):
        split = 4 * len(seqs) // 5
        if train:
            indices = slice(0, split)
        else:
            indices = slice(split, -1)

        self.sequences = seqs[indices]
        self.deltas = deltas[indices]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].float()
        sequence = torch.transpose(sequence, 0, 1)

        return sequence, self.deltas[idx].float()


def get_train_and_validate_datasets(experiment_name, mers=1):
    experiment_dict = {
        "ets1_ets1":
            {
                "labeled_data_path": "./data/lbled_o1_selected.csv",
                "training_data_path": "./data/train_ets1_ets1.tsv",
            },
        "ets1_runx1":
            {
                "labeled_data_path": "./data/both_ori_plt_ets1_runx1.csv",
                "training_data_path": "./data/train_ets1_runx1.tsv",
            },
    }

    experiment = experiment_dict[experiment_name]

    df_delta = pd.read_csv(experiment["labeled_data_path"])
    dft = pd.read_csv(experiment["training_data_path"], sep="\t")

    df_delta["delta"] = df_delta["indiv_median"]  # df_delta["two_median"] - df_delta["indiv_median"]

    dft = dft.merge(df_delta, on="Name")

    dft = dft.sample(frac=1).reset_index(drop=True)

    # this line and the next enable/disable filtering on "cooperative" classification
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
    scaler = StandardScaler().fit(deltas)  # need to only apply this to the input
    deltas = scaler.transform(deltas)
    deltas = np.squeeze(deltas, axis=1)
    deltas = torch.from_numpy(deltas)

    return (SequenceDataset(sequences, deltas, train=True),
            SequenceDataset(sequences, deltas, train=False))
