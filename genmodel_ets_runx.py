import os
import pandas as pd
import pickle
from sklearn import ensemble, tree
import subprocess
import argparse

from coopgcpbm.modeler.cooptrain import CoopTrain
from coopgcpbm.modeler.bestmodel import BestModel
import coopgcpbm.modeler.plotlib as pl

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

    rf_param_grid = {
        'n_estimators': [500,750,1000],
        'max_depth': [5,10,15],
        "min_samples_leaf": [5,10,15],
        "min_samples_split" : [5,10,15]
    }

    best_models = {
        "strength":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "affinity": {"colnames": ("ets1_score","runx1_score")}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "distance":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "orientation":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "orientation": {"relative":False, "pos_cols": {"ets1_pos":"ets1_ori", "runx1_pos":"runx1_ori"}}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "strength,orientation":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "affinity": {"colnames": ("ets1_score","runx1_score")},
                    "orientation": {"relative":False, "pos_cols": {"ets1_pos":"ets1_ori", "runx1_pos":"runx1_ori"}}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "strength,distance":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"},
                    "affinity": {"colnames": ("ets1_score","runx1_score")}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "strength,distance,orientation":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"},
                    "affinity": {"colnames": ("ets1_score","runx1_score")},
                    "orientation": {"relative":False, "pos_cols": {"ets1_pos":"ets1_ori", "runx1_pos":"runx1_ori"}}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
    }

    pl.plot_model_metrics(best_models, path="auc.png", cvfold=10, score_type="auc", varyline=True, title="Average ROC Curves for Ets1-Runx1", interp=True)

    feature_dict = {
        "distance":{"type":"numerical"},
        "affinity": {"colnames": ("ets1_score","runx1_score")},
        "orientation": {"relative":False, "pos_cols": {"ets1_pos":"ets1_ori", "runx1_pos":"runx1_ori"}}
    }
    train = ct.get_feature_all(feature_dict)
    label = ct.get_numeric_label({'cooperative': 1, 'independent': 0})
    rf = best_models["strength,distance,orientation"][1]
    rf.fit(train,label)
    model_name = "ets1_runx1_rfmodel.sav"
    pickle.dump(rf, open(model_name, 'wb'))
    print("Model saved in %s" % model_name)

    # tree.export_graphviz(m.estimators_[5], out_file='tree.dot',
    #         feature_names = train.columns,
    #         class_names = ['additive','cooperative'],
    #         rounded = True, proportion = False,
    #         precision = 2, filled = True)
    # subprocess.call(['dot', '-Tpdf', 'tree.dot', '-o', 'tree.pdf', '-Gdpi=600'])
