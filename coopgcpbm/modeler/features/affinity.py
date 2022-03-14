
from chip2probe.modeler.features import basefeature

class Affinity(basefeature.BaseFeature):
    def __init__(self, traindf, params):
        """
        Affinity prediction feature class

        Args:
            traindf: dataframe containing the site_wk_score and site_str_score columns
                     if imadsmodel is None
            params:
                - imads

         Returns:
            NA
        """
        default_args = {
            "imads" : False,
            "colnames":("site_wk_score", "site_str_score")
        }
        self.df = traindf
        self.set_attrs(params, default_args)

        if len(self.colnames) > 2 or len(self.colnames) == 0:
            raise Exception("there should be only 1 or 2 columns")
        if len(self.colnames) == 1:
            self.col1, self.col2 = self.colnames[0], None
        else:
            self.col1, self.col2 = self.colnames[0], self.colnames[1]

    # TODO: make prediction can do without str weak
    def get_feature(self,seqcolname="Sequence"):
        """Get a dictionary of binding site preference scores."""
        rfeature = []
        # if imadsmodel is not provided, we assume the weak and strong site position is provided
        # TODO: make general column name
        for idx, row in self.df.iterrows():
            if self.imads: # imads model is provided
                pr = self.imads.predict_sequence(row[seqcolname])
                if len(pr) != 2:
                    #print("Found sites more than 2 sites, using site position in the table as reference")
                    newpr = []
                    # if we suddenly have > 2 predictions, try using position in the training
                    for p in pr:
                        if p["core_mid"] == row["%s_pos"%self.col2] or p["core_mid"] == row["%s_pos"%self.col1]:
                            newpr.append(p)
                    if len(newpr) != 2:
                        print("Error on row:\n", row, "with prediction\n", pr)
                        raise Exception("Sequence does not have 2 binding sites")
                    pr = newpr
                f = {"site_wk_score": pr[0]["score"] if pr[0]["score"] < pr[1]["score"] else pr[1]["score"],
                     "site_str_score": pr[0]["score"] if pr[0]["score"] > pr[1]["score"] else pr[1]["score"]}
            else:
                f = {self.col1: row[self.col1]} if self.col2 == None else {self.col1: row[self.col1], self.col2: row[self.col2]}
            rfeature.append(f)
        return rfeature
