import os
import argparse
import pandas as pd
from sklearn import ensemble, tree
import subprocess
import pickle

from coopgcpbm.modeler.cooptrain import CoopTrain
from coopgcpbm.modeler.bestmodel import BestModel
import coopgcpbm.modeler.plotlib as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Generate ETS1-ETS1 random forest models')
    parser.add_argument(action="store", dest="path", type=str, help='Training file')
    parser.add_argument('-o', '--outdir', action="store", dest="outdir", default=".", help='output directory to store output files')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    df = pd.read_csv(args.path,sep="\t")
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
                    "affinity": {"colnames": ("site_str_score","site_wk_score")}
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
                    "orientation": {"relative":True, "one_hot":True, "pos_cols": {"site_str_pos":"site_str_ori", "site_wk_pos":"site_wk_ori"}}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "distance,orientation":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"},
                    "orientation": {"relative":True, "one_hot":True, "pos_cols": {"site_str_pos":"site_str_ori", "site_wk_pos":"site_wk_ori"}}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "strength,distance":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"},
                    "affinity": {"colnames": ("site_str_score","site_wk_score")}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "strength,distance,orientation":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"},
                    "affinity": {"colnames": ("site_str_score","site_wk_score")},
                    "orientation": {"relative":True, "one_hot":True, "pos_cols": {"site_str_pos":"site_str_ori", "site_wk_pos":"site_wk_ori"}}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all()
    }

    pl.plot_model_metrics(best_models, path=os.path.join(args.outdir, "auc_all.png"), cvfold=10, score_type="auc", varyline=True, title="Average ROC Curves for ETS1-ETS1", interp=True)

    rf = best_models["strength,distance,orientation"][1]

    train = ct.get_feature_all({
        "distance":{"type":"numerical"},
        "affinity": {"colnames": ("site_str_score","site_wk_score")},
        "orientation": {"relative":True, "one_hot":True, "pos_cols": {"site_str_pos":"site_str_ori", "site_wk_pos":"site_wk_ori"}}
    })
    label = ct.get_numeric_label({'cooperative': 1, 'independent': 0})

    rf.fit(train,label)
    model_name = os.path.join(args.outdir, "ets1_ets1_rfmodel.sav")
    pickle.dump(rf, open(model_name, 'wb'))
    print("Model saved in %s" % model_name)
