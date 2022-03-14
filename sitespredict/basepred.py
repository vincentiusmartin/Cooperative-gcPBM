'''
Created on Jul 26, 2019

@author: vincentiusmartin
'''


class BasePrediction():
    '''
    classdocs
    '''


    def __init__(self, sequence, prediction_dict):
        '''
        Constructor
        '''
        self.sequence = sequence
        self.predictions = prediction_dict
    
    def __str__(self):
        return "sequence: %s\npredictions: %s" % (self.sequence, str(self.predictions)) 
        
    def get_predictions_key(self):
        return self.prediction.keys()