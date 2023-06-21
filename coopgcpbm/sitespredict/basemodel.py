'''
Created on Jul 16, 2019

@author: vincentiusmartin
'''

import abc
import pandas as pd
from coopgcpbm.util import bio as bio

class BaseModel(abc.ABC):
    '''
    classdocs
    '''

    @abc.abstractclassmethod
    def __init__(self):
        pass

    def pred_input_todict(self, sequence_input, sequence_colname="sequence",
                          key_colname="", predict_flanks=True):
        """
        Get the dictionary form of the input for sequences predictions.

        sequence_input types allowed: Datafram, dictionary
        """
        # check if input is a dataframe
        if isinstance(sequence_input, pd.DataFrame):
            return bio.get_seqdict(sequence_input,
                                   sequence_colname=sequence_colname,
                                   keycolname=key_colname)
        # check if input is a dictionary
        elif isinstance(sequence_input, dict):
            return sequence_input
        # raise exception if input type is not allowed
        else:
            raise Exception("input must be data frame or dictionary of sequences")

    @abc.abstractclassmethod
    def predict_sequences(self, sequence_df):
        """
        Should return basepred
        """
        pass

    @abc.abstractclassmethod
    def make_plot_data(self, sequence_df):
        pass
