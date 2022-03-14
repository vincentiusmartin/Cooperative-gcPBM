'''
Created on Oct 30, 2019

@author: vincentiusmartin
@editedby: Farica Zhuang
'''

import pandas as pd
import sys
import importlib

from chip2probe.util import util
from chip2probe.util import bio
from chip2probe.modeler.features import *

class CoopTrain:
    def __init__(self, trainingdata, corelen=0, sep="\t", flip_th=False,
                positive_cores=[], imads=None, ignore_sites_err=False, seqcolname="Sequence"):
        """
        Cooperative Training class constructor

        Args:
            traininngdata : Path string or data frame with the required columns
            corelen : The length of the core binding site
            sep: Separator for the csv/tsv file if the trainingdata is string
            flip_th: Flip all TH orientation to HT (important whe we want to treat TH == HT)
            positive_cores: Required if flip_th is true, contain all the positive cores that define
            HT orientation.
            imads: if the input is list, need to give imads model
            ignore_sites_err: if True, skip sequences with sites not equal to 2
        Returns:
            NA
        """
        self.ignore_sites_err = ignore_sites_err
        self.corelen = corelen
        self.seqcolname = seqcolname
        # Get the positive cores
        if imads: # take positive cores from imads
            positive_cores = [m.core for  m in imads.models]
        elif (flip_th or isinstance(trainingdata, list)) and not positive_cores:
            raise Exception("Positive cores must be defined when flip_th is True")
        if isinstance(trainingdata, pd.DataFrame):
            self.df = trainingdata.copy().reset_index(drop=True)
        elif isinstance(trainingdata, str):
            self.df = pd.read_csv(trainingpath, sep=sep)
        elif isinstance(trainingdata, list): # input is list of string
            if imads is None:
                raise Exception("imads must be provided when input is list")
            self.df = self.make_train(trainingdata, imads)
            ori = self.get_feature("orientation", {"positive_cores":positive_cores, "relative":True, "one_hot":False})
            self.df["orientation"] = [x["ori"] for x in ori]
        else:
            raise Exception("input must be string path or data frame")
        #self.validate_cols()
        if flip_th:
            self.df = self.flip_th2ht(positive_cores)

    def make_train(self, seqs, imads):
        trainlist = []
        idx = 1
        for seq in seqs:
            row = {}
            # need to put index first so it is not affected by the sequences we ignore
            row["name"] = "seq-%d" % idx
            idx += 1
            preds = imads.predict_sequence(seq)
            if len(preds) != 2:
                if self.ignore_sites_err:
                    continue
                raise Exception("Number of sites must be 2 but found %d for %s" % (len(preds), seq))
            row["sequence"] = seq
            row["distance"] = preds[1]["core_mid"] - preds[0]["core_mid"]
            if preds[0]["score"] > preds[1]["score"]:
                row["site_wk_score"] = preds[1]["score"]
                row["site_wk_pos"] = preds[1]["core_mid"]
                row["site_str_score"] = preds[0]["score"]
                row["site_str_pos"] = preds[0]["core_mid"]
            else:
                row["site_wk_score"] = preds[0]["score"]
                row["site_wk_pos"] = preds[0]["core_mid"]
                row["site_str_score"] = preds[1]["score"]
                row["site_str_pos"] = preds[1]["core_mid"]
            trainlist.append(row)
        return pd.DataFrame(trainlist)

    def validate_cols(self):
        """
        Check if we have all required columns.
        """
        required_cols = {"label"}
        if not required_cols.issubset(self.df.columns):
            raise Exception("Missing columns, required columns are: {}".format(required_cols))

    def flip_th2ht(self,positive_cores):
        # flip if orientation one
        ori = self.get_feature("orientation", {"positive_cores":positive_cores, "relative":True, "one_hot":False})
        records = self.df.to_dict(orient='records')
        for i in range(0,len(records)):
            if ori[i]["ori"] == "HT/TH":
                site_str = records[i]["sequence"][records[i]["site_str_pos"] - self.corelen//2:records[i]["site_str_pos"] + self.corelen//2]
                site_wk = records[i]["sequence"][records[i]["site_wk_pos"] - self.corelen//2:records[i]["site_wk_pos"] + self.corelen//2]
                if site_str not in positive_cores and site_wk not in positive_cores:
                    records[i]["sequence"] = bio.revcompstr(records[i]["sequence"])
                    # flip the position as well
                    str_pos = records[i]["site_str_pos"]
                    wk_pos = records[i]["site_wk_pos"]
                    records[i]["site_str_pos"] = len(records[i]["sequence"]) - wk_pos
                    records[i]["site_wk_pos"] = len(records[i]["sequence"]) - str_pos
        return pd.DataFrame(records)

    # ---------- GETTING FEATURE ----------
    def get_feature(self, feature, params):
        """
        Get a feature

        Args:
            feature: a string, the name of the feature (e.g. "orientation")
         Returns:
            feature_dict
        """

        module = importlib.import_module("chip2probe.modeler.features.%s" % feature)
        class_ = getattr(module, feature.capitalize())
        instance = class_(self.df, params)
        return instance.get_feature(seqcolname=self.seqcolname)

    def get_feature_all(self, feature_dict, rettype="df"):
        """
        Get all feature based on feature dict

        Accepts a dictionary of feature name and parameters relative to the feature.
        All features are from the ``chip2probe.modeler.features`` package. If more
        than one of the same feature are needed, then add description to the name
        separated by underscore (e.g. shape_in, shape_out)

        Args:
            feature_dict: the following is the list of currently available feature:
                1. distance: {type:"numeric/categorical"}
                2. orientation: {"positive_cores:[]", relative:Bool, one_hot:Bool}
                3. affinity: {imads: sitespredict.imads instance}
                4. shape: {"seqin": int, "smode": "positional/strength", "direction": "inout/orientation", "positive_cores" : []}
                5. sequence:
            rettype: dict, list, df
         Returns:
            ldict: list of dictionary of features or list of list if 'aslist' is True
        """
        ldict = []
        for class_name, params in feature_dict.items():
            # separate class by underscore, the first entry should always be the feature name
            cname = class_name.split("_")[0]
            ldict = util.merge_listdict(self.get_feature(cname,params),ldict)
        if rettype == "list":
            # return as list of list
            return [[d[k] for k in d] for d in ldict]
        elif rettype == "dict":
            return ldict
        else:
            return pd.DataFrame(ldict)

    def get_numeric_label(self, label_map):
        ytrain = self.df['label'].map(label_map)
        return ytrain

    def get_training_df(self, feature_dict, label_map={'cooperative': 1, 'additive': 0}):
        """Get training df from a dictionary of features."""
        ldict = self.get_feature_all(feature_dict)
        train = pd.DataFrame(ldict)
        train['label'] = self.get_numeric_label(label_map).values
        return train
