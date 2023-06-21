import pickle
from coopgcpbm.modeler.cooptrain import CoopTrain
import pandas as pd

class ShapeModel:
    def __init__(self, paramdict):
        self.model = {}
        self.param = {}
        for key, val in paramdict.items():
            self.model[key] = pickle.load(open(val["path"], "rb"))
            self.param[key] = val["param"]

    def predict(self, df):
        # hardcoded for now, too tired
        print(df)
        predres = {}
        predproba = {}
        for key in self.model:
            curdf = df.loc[df["orientation"] == key]

            curct = CoopTrain(curdf, corelen=4, flip_th=True, positive_cores=["GGAA","GGAT"])
            feature_dict = {
                    "distance":{"type":"numerical"},
                    "shape_in": {"seqin":4, "smode":"positional", "direction":"inout"}, # maximum seqin is 4
                    "shape_out": {"seqin":-4, "smode":"positional", "direction":"inout"}
                }
            train_df = pd.DataFrame(curct.get_feature_all(feature_dict))[self.param[key]]
            train = train_df.values.tolist()
            pred = self.model[key].predict(train)
            proba = [prb[1] for prb in self.model[key].predict_proba(train)]
            idxs = curdf.index
            for i in range(0,len(pred)):
                predres[idxs[i]] = pred[i]
                predproba[idxs[i]] = proba[i]
        allidxs = sorted(predres.keys())
        predlist = [predres[i] for i in allidxs]
        probalist = [predproba[i] for i in allidxs]
        return predlist, probalist
