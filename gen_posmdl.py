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
    parser = argparse.ArgumentParser(description = 'Generate models trained using sequence+shape features')
    parser.add_argument(action="store", dest="path", type=str, help='File input path')
    parser.add_argument('-a', '--s1', action="store", dest="s1", type=str,  help='Column name for the first TF site')
    parser.add_argument('-b', '--s2', action="store", dest="s2", type=str,  help='Column name for the second TF site')
    parser.add_argument('-s', '--smode', action="store", dest="smode", type=str,  help='Site mode: positional/relative')
    parser.add_argument('-r', '--rel_ori',  action="store_true", dest="rel_ori", help='Use relative orientations')
    parser.add_argument('-o', '--one_hot_ori',  action="store_true", dest="one_hot_ori", help='Represent orientation feature in one hot encoding')
    args = parser.parse_args()

    # basepath = "output/Ets1Runx1"
    # trainingpath = "%s/training/train_ets1_runx1.tsv" % basepath
    # s1, s2 = "ets1", "runx1"
    # rel_ori = False
    # one_hot_ori = False
    # smode = "positional"

    # basepath = "output/Ets1Ets1_v2"
    # trainingpath = "%s/training/train_ets1_ets1.tsv" % basepath
    # s1, s2 = "site_str", "site_wk"
    # rel_ori = True
    # one_hot_ori = True
    # smode = "relative"

    s1,s2 = args.s1, args.s2
    smode = args.smode
    rel_ori = args.rel_ori
    one_hot_ori = args.one_hot_ori

    df = pd.read_csv(args.path, sep="\t")
    ct = CoopTrain(df)
    pd.set_option("display.max_columns",None)

    rf_param_grid = {
        'n_estimators': [500],#[500,750,1000],
        'max_depth': [10],#[5,10,15],
        "min_samples_leaf": [10] ,#[5,10,15],
        "min_samples_split" :  [10]#[5,10,15]
    }

    best_models = {
        "distance,orientation":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"},
                    "orientation": {"relative":rel_ori, "one_hot":one_hot_ori, "pos_cols": {"%s_pos"%s1:"%s_ori"%s1, "%s_pos"%s2:"%s_ori"%s2}},
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "distance,orientation,sequence":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"},
                    "orientation": {"relative":rel_ori, "one_hot":one_hot_ori, "pos_cols": {"%s_pos"%s1:"%s_ori"%s1, "%s_pos"%s2:"%s_ori"%s2}},
                    "sequence_in":{"seqin":3, "poscols":['%s_pos'%s1,'%s_pos'%s2], "namecol":"Name", "smode":smode, 'maxk':1},
                    "sequence_out":{"seqin":-4, "poscols":['%s_pos'%s1,'%s_pos'%s2], "namecol":"Name", "smode":smode, 'maxk':1}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "distance,orientation,shape":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"},
                    "orientation": {"relative":rel_ori, "one_hot":one_hot_ori, "pos_cols": {"%s_pos"%s1:"%s_ori"%s1, "%s_pos"%s2:"%s_ori"%s2}},
                    "shape_in":{"seqin":3, "poscols":['%s_pos'%s1,'%s_pos'%s2], "smode":smode},
                    "shape_out":{"seqin":-4, "poscols":['%s_pos'%s1,'%s_pos'%s2], "smode":smode}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "distance,orientation,sequence,shape":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"},
                    "orientation": {"relative":rel_ori, "one_hot":one_hot_ori, "pos_cols": {"%s_pos"%s1:"%s_ori"%s1, "%s_pos"%s2:"%s_ori"%s2}},
                    "shape_in":{"seqin":3, "poscols":['%s_pos'%s1,'%s_pos'%s2], "smode":smode},
                    "shape_out":{"seqin":-4, "poscols":['%s_pos'%s1,'%s_pos'%s2], "smode":smode},
                    "sequence_in":{"seqin":3, "poscols":['%s_pos'%s1,'%s_pos'%s2], "namecol":"Name", "smode":smode},
                    "sequence_out":{"seqin":-4, "poscols":['%s_pos'%s1,'%s_pos'%s2], "namecol":"Name", "smode":smode}
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
        "distance,orientation,strength":
            BestModel(clf="sklearn.ensemble.RandomForestClassifier",
              param_grid=rf_param_grid,
              train_data=ct.get_training_df({
                    "distance":{"type":"numerical"},
                    "affinity": {"colnames": ("%s_score"%s1,"%s_score"%s2)},
                    "orientation": {"relative":rel_ori, "one_hot":one_hot_ori, "pos_cols": {"%s_pos"%s1:"%s_ori"%s1, "%s_pos"%s2:"%s_ori"%s2}},
                }, label_map={'cooperative': 1, 'independent': 0})
            ).run_all(),
    }

    pl.plot_model_metrics(best_models, path="auc_posfeatures.pdf", cvfold=10, score_type="auc", varyline=True, title="AUC Shape features", interp=True)

    feature_dict = {
        "distance":{"type":"numerical"},
        "orientation": {"relative":rel_ori, "one_hot":one_hot_ori, "pos_cols": {"%s_pos"%s1:"%s_ori"%s1, "%s_pos"%s2:"%s_ori"%s2}},
        "shape_in":{"seqin":3, "poscols":['%s_pos'%s1,'%s_pos'%s2], "smode":smode},
        "shape_out":{"seqin":-4, "poscols":['%s_pos'%s1,'%s_pos'%s2], "smode":smode},
        "sequence_in":{"seqin":3, "poscols":['%s_pos'%s1,'%s_pos'%s2], "namecol":"Name", "smode":smode},
        "sequence_out":{"seqin":-4, "poscols":['%s_pos'%s1,'%s_pos'%s2], "namecol":"Name", "smode":smode}
    }
    train = ct.get_feature_all(feature_dict)
    label = ct.get_numeric_label({'cooperative': 1, 'independent': 0})
    rf = best_models["distance,orientation,sequence,shape"][1]
    rf.fit(train,label)
    model_name = "rfposmodel.sav"
    pickle.dump(rf, open(model_name, 'wb'))
    print("Model saved in %s" % model_name)
