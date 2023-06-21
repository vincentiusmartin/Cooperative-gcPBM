
import pandas as pd

from coopgcpbm.modeler.features import basefeature

class Distance(basefeature.BaseFeature):
    def __init__(self, traindf, params):
        # TODO: can incorporte site model to detect distance automatically
        """
        Distance feature class

        Args:
            traindf: dataframe containing the "distance" column
            params:
                - type: numerical / categorial representation of distance

         Returns:
            NA
        """
        default_args = {
            "type" : "numerical"
        }
        self.df = traindf
        self.set_attrs(params, default_args)

    def get_feature(self,seqcolname="Sequence"):
        """
        Return a list of dictionaries for distance feature.
        Key is the column name and value is the column value
        """
        if self.type == "numerical":
            return [{"dist_numeric":x} for x in self.df["distance"].values]
        elif self.type == "categorical":
            # get one-hot encoded version of distance as a dataframe
            one_hot = pd.get_dummies(self.df['distance'])
            # rename the columns
            one_hot.columns = ["dist_cat_%d"%col for col in one_hot.columns]
            # return as a list of dictionaries
            return one_hot.to_dict('records')
        else:
            raise Exception("distance must be numerical or categorical")
