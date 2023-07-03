import os
import pandas as pd
import pickle
from sklearn import ensemble, tree
import subprocess
import argparse

import sys
sys.path.append("../../..")

from coopgcpbm.modeler.cooptrain import CoopTrain
from coopgcpbm.modeler.bestmodel import BestModel
import coopgcpbm.modeler.plotlib as pl

# python3 genmodel_ets_runx_test.py /Users/vincentiusmartin/Research/chip2gcPBM/Cooperative-gcPBM/data/analysis_files/ETS1_RUNX1/training/train_ets1_runx1.tsv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Generate ETS1-RUNX1 random forest models')
    parser.add_argument(action="store", dest="path", type=str, help='Training file')
    args = parser.parse_args()

    # basepath = "output/Ets1Runx1"
    # trainingpath = "output/Ets1Runx1/training/train_ets1_runx1.tsv"

    # basepath = "output/Runx1Ets1"
    # trainingpath = "%s/training/train_runx1_ets1.tsv" % basepath

    print(args.path)
    df = pd.read_csv(args.path, sep="\t")
    ct = CoopTrain(df)

    rf_param_grid1 = {
        'max_depth': [1]
    }
    rf_param_grid2 = {
        'max_depth': [2]
    }
    rf_param_grid3 = {
        'max_depth': [3]
    }
    rf_param_grid4 = {
        'max_depth': [4]
    }
    rf_param_grid5 = {
        'max_depth': [5]
    }
    rf_param_grid6 = {
        'max_depth': [6]
    }
    rf_param_grid7 = {
        'max_depth': [7]
    }
    rf_param_grid8 = {
        'max_depth': [8]
    }

    best_models = {
        "max_depth=1":
            BestModel(clf="sklearn.tree.DecisionTreeClassifier",
              param_grid=rf_param_grid1,
              train_data=ct.get_training_df({
                    "affinity": {"colnames": ["runx1_score"]}
                    }
                , label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "max_depth=2":
            BestModel(clf="sklearn.tree.DecisionTreeClassifier",
              param_grid=rf_param_grid2,
              train_data=ct.get_training_df({
                    "affinity": {"colnames": ["runx1_score"]}
                    }
                , label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "max_depth=3":
            BestModel(clf="sklearn.tree.DecisionTreeClassifier",
              param_grid=rf_param_grid3,
              train_data=ct.get_training_df({
                    "affinity": {"colnames": ["runx1_score"]}
                    }
                , label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "max_depth=4":
            BestModel(clf="sklearn.tree.DecisionTreeClassifier",
              param_grid=rf_param_grid4,
              train_data=ct.get_training_df({
                    "affinity": {"colnames": ["runx1_score"]}
                    }
                , label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "max_depth=5":
            BestModel(clf="sklearn.tree.DecisionTreeClassifier",
              param_grid=rf_param_grid5,
              train_data=ct.get_training_df({
                    "affinity": {"colnames": ["runx1_score"]}
                    }
                , label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "max_depth=6":
            BestModel(clf="sklearn.tree.DecisionTreeClassifier",
              param_grid=rf_param_grid6,
              train_data=ct.get_training_df({
                    "affinity": {"colnames": ["runx1_score"]}
                    }
                , label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "max_depth=7":
            BestModel(clf="sklearn.tree.DecisionTreeClassifier",
              param_grid=rf_param_grid7,
              train_data=ct.get_training_df({
                    "affinity": {"colnames": ["runx1_score"]}
                    }
                , label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "max_depth=8":
            BestModel(clf="sklearn.tree.DecisionTreeClassifier",
              param_grid=rf_param_grid8,
              train_data=ct.get_training_df({
                    "affinity": {"colnames": ["runx1_score"]}
                    }
                , label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
    }

    pl.plot_model_metrics(best_models, path="auc.png", cvfold=10, score_type="auc", varyline=True, title="Decision tree performance on Ets1-Runx1\nusing only Runx1 binding strength", interp=True)

    feature_dict = {
        "distance":{"type":"numerical"},
        "affinity": {"colnames": ("ets1_score","runx1_score")},
        "orientation": {"relative":False, "pos_cols": {"ets1_pos":"ets1_ori", "runx1_pos":"runx1_ori"}}
    }
    train = ct.get_feature_all(feature_dict)
    label = ct.get_numeric_label({'cooperative': 1, 'independent': 0})
    rf = best_models["max_depth=5"][1]
    rf.fit(train,label)
    model_name = "ets1_runx1_rfmodel.sav"
    pickle.dump(rf, open(model_name, 'wb'))
    print("Model saved in %s" % model_name)

    tree.export_graphviz(rf, out_file='tree.dot',
            feature_names = train.columns,
            class_names = ['independent','cooperative'],
            rounded = True, proportion = False,
            precision = 2, filled = True)
    subprocess.call(['dot', '-Tpdf', 'tree.dot', '-o', 'tree.pdf', '-Gdpi=600'])
