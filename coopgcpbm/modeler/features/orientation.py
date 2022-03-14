import pandas as pd

from chip2probe.util import bio
from chip2probe.modeler.features import basefeature


class Orientation(basefeature.BaseFeature):
    def __init__(self, traindf, params):
        """
        Orientation feature class

        Args:
            traindf: dataframe containing the sequence column
            params:
                - positive_cores: list of string of positive cores
                - relative: use site orientation relative to each other (HH, HT/TH, TT)
                - one_hot: use one_hot representation
                - pos_cols: the name of position columns. If there is only one tf,
                    use list. If the position and ori is known, use dictionary;
                    e.g. {"ets_pos":"ets_ori", "runx_pos":"runx_ori"}
         Returns:
            NA
        """

        default_args = {
            "positive_cores":[],
            "relative": True,
            "one_hot": False,
            "pos_cols": ("site_wk_pos", "site_str_pos"),
        }
        self.df = traindf
        self.set_attrs(params, default_args)
        self.motiflen = len(self.positive_cores[0]) if self.positive_cores else 0

        # if pos cols is dict then we have orientation already
        if isinstance(self.pos_cols, dict):
            pc = [*self.pos_cols] # get only the keys
            self.ori_cols = [self.pos_cols[k] for k in pc]
            self.pos_cols = pc
        else: # then we need to determine ori from positive cores
            self.ori_cols = []

    def get_feature(self, seqcolname="Sequence"):
        """

        """
        negative_cores = [bio.revcompstr(p) for p in self.positive_cores]
        rfeature = []
        for idx,row in self.df.iterrows():
            # get the binding sites
            if row[self.pos_cols[0]] < row[self.pos_cols[1]]:
                site1, site2 = row[self.pos_cols[0]], row[self.pos_cols[1]]
            else:
                site1, site2 = row[self.pos_cols[1]], row[self.pos_cols[0]]
            if self.ori_cols: # we know the orientation already
                if row[self.pos_cols[0]] < row[self.pos_cols[1]]:
                    s1, s2 = row[self.ori_cols[0]], row[self.ori_cols[1]]
                else:
                    s1, s2 = row[self.ori_cols[1]], row[self.ori_cols[0]]
            else:
                seq = row[seqcolname]
                p1 = seq[site1 - self.motiflen//2:site1 + self.motiflen//2]
                p2 = seq[site2 - self.motiflen//2:site2 + self.motiflen//2]
                if p1 in self.positive_cores:
                    s1 = 1
                elif p1 in negative_cores:
                    s1 = 0
                else:
                    s1 = -999
                    print("couldn't find the first site %s in %s in the core list" % (p1,seq))
                if p2 in self.positive_cores:
                    s2 = 1
                elif p2 in negative_cores:
                    s2 = 0
                else:
                    s2 = -999
                    print("couldn't find the second site %s in %s in the core list" % (p2,seq))
            if self.relative:
                # -/+ == TT, +- == HH
                if s1 == s2:
                    ori = '+/+'
                elif s1 == 1 and s2 == 0:
                    ori = '+/-'
                elif s1 == 0 and s2 == 1:
                    ori = '-/+'
                else:
                    ori = '-1'
                rfeature.append({"ori":ori})
            if not self.relative:
                rfeature.append({"ori1":s1, "ori2":s2})
        if self.relative and self.one_hot:
            dum_df = pd.DataFrame(rfeature)
            notfound = {"+/+","+/-","-/+"} - set(dum_df["ori"].unique())
            dum_rec = pd.get_dummies(dum_df).to_dict('records')
            for nf in notfound:
                print("notfound orientation in the dataset: ",nf)
                for i in range(len(dum_rec)):
                    dum_rec[i]["ori_%s" % nf] = 0
            return dum_rec
        else:
            return rfeature
