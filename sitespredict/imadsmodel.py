"""
This file contains iMADSModel class.

Created on Jul 16, 2019

Authors: Vincentius Martin, Farica Zhuang

This file is a modified version of https://github.com/Duke-GCB/Predict-TF-Binding

Download libsvm:
https://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/index-1.0.html
Set up libsvm:
Run make in libsvm-3.24 directory
Run make in libsvm-3.24/python directory
"""

import os
import sys
from libsvm import svmutil


class iMADSModel(object):
    '''
    classdocs
    '''


    def __init__(self, model_path, core, width, kmers):
        '''
        Constructor
        '''

        filename = os.path.basename(model_path)
        print("Loading imads model: %s" % filename)
        if not os.path.exists(model_path):
            raise Exception("File %s doesn't exist" % model_path)
        self.modeldict = self.load_model(model_path)

        self.core = core
        self.width = width
        self.kmers = kmers

        # TODO:
        # if either of these params is None then check from path
        # if core is None:
        #    self.core = int(re.search("transformed\_(.*)bp",fname).group(1))


    def load_model(self, model_file, check_size=True):
        """
        Taken from: https://github.com/Duke-GCB/Predict-TF-Binding
        Loads a svm model from a file and computes its size
        :param model_file: The file name of the model to load
        :return: A dictionary with keys model, file, and size
        """
        model = svmutil.svm_load_model(model_file)
        model_dict = {'model': model}
        if check_size:
            # with >= libsvm-3.24, use this
            model_dict['size'] = len(model.get_SV()[0])
            # with < libsvm-3.24, use this
            # model_dict['size'] = len(model.get_SV()[0]) - 1 # sv includes a -1 term that is not present in the model file, so subtract 1
        return model_dict

    def predict(self, features, const_intercept=False):
        """
        Run prediction using svm_predict.
        :param features: List of features, produced by svr_features_from_sequence
        :param model: A loaded svm model (from load_model)
        :param const_intercept: if true, add a 1:1 term at the beginning of the matrix. Must match model's term
        :return: triple of predictions, accuracy, and values from svm_predict.
        """
        feature_size = len(features)
        if const_intercept:
            feature_size += 1 # If we are to use a const intercept term, we will have one more feature
        #if 'size' in self.modeldict and self.modeldict['size'] != feature_size:
        #    # vm Exception: Model size 1536 does not match feature size 384.
        #    raise Exception('Model size {} does not match feature size {}.\nPlease check parameters for width, '
        #                    'kmers, and const_intercept'.format(self.modeldict['size'], feature_size))
        svm_matrix = dict()
        # Build the dictionary that corresponds to the matrix file
        offset = 1 # svm_matrix is a dictionary of index to value, starting at 1
        if const_intercept:
            svm_matrix[offset] = 1
            offset += 1
        for i, feature in enumerate(features):
            svm_matrix[i + offset] = feature['value']
        predictions = svmutil.svm_predict([1.0], [svm_matrix], self.modeldict['model'], '-q')
        return predictions
