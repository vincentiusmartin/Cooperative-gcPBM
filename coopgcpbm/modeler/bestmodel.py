'''
1. Use all features, find best param
2. find the best top n
3. use those features  to find best param again
'''
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
import itertools
import concurrent.futures as cc
import timeit
import os, sys
import functools
import importlib
import six

## TODO:
# 1. refactor code: no self assignment on the loop outside constructor
# 2. write log to file
class BestModel:
    def __init__(self, clf, param_grid, train_data, topn = -1, cv_fold=10, cv_num=1):
        """
        Get best model

        Get the best models from the given grid search on param_grid

        Args:
            clf: a string of machine learning model to use, e.g. "sklearn.ensemble.RandomForestClassifier"
            param_grid: a dictionary of the grid search parameters
            train_data: a data frame of features to use, can be obtained using 'get_training_df'
                    in CoopTrain class.
            topn: Pick top n features to use.
            cv_fold: number of cross validation fold
            cv_num: number of cross validation run to average

        Returns:
            (NA)
        """
        # getattr(sys.modules[__name__], "sklearn.ensemble.RandomForestClassifier")
        self.classifier = self.import_string(clf)
        self.param_grid = param_grid
        self.topn = topn # integer
        self.train_data = train_data

        if self.topn == -1:
            self.topn = len(train_data.columns)-1

        self.cv_fold = cv_fold
        self.cv_num = cv_num

    def import_string(self, dotted_path):
        """
        Import a dotted module path and return the attribute/class designated by the
        last name in the path. Raise ImportError if the import failed.

        From: https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object
        """
        try:
            module_path, class_name = dotted_path.rsplit('.', 1)
        except ValueError:
            msg = "%s doesn't look like a module path" % dotted_path
            six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])

        module = importlib.import_module(module_path)

        try:
            return getattr(module, class_name)
        except AttributeError:
            msg = 'Module "%s" does not define a "%s" attribute/class' % (
                module_path, class_name)
            six.reraise(ImportError, ImportError(msg), sys.exc_info()[2])

    def run_all(self, num_workers=os.cpu_count(), score_type="auc"):
        """
        Get best model

        Get the best models from the given grid search on param_grid

        Args:
            num_workers
            score_type

        Returns:
            (NA)
        """
        # initialize classifier
        self.init_data(self.train_data)
        # get best hyperparam combination
        clf = self.get_best_param(num_workers=num_workers,score=score_type)
        new_x = self.train_data
        # get best top n, can only get topn if we have more features than 'n'
        if self.topn < len(self.train_data.columns) - 1:
            new_x = self.get_topn(clf)
            # get best hyperparam for top n
            clf = self.get_best_param(num_workers=num_workers,score=score_type)
        # return the best model for top n
        return new_x, clf

    def init_data(self, df):
        self.x_train = df.loc[:, df.columns != 'label']
        self.y_train = df.loc[:, 'label']

    def run_kfold(self, comb, score="auc"):
        """
        metrics: auc (Area Under ROC Curve)/pr (Precision Recall)
        """
        x_train = self.x_train.values
        y_train = self.y_train.values
        keys  = list(self.param_grid.keys())

        comb_dict = {keys[i]:comb[i] for i in range(len(keys))}
        clf = self.classifier(**comb_dict)
        avg_acc = 0
        avg_scr = 0
        # take the average of runs
        for i in range(self.cv_num):
            # perform 10 fold cross validation
            tot_acc = 0
            tot_score = 0
            cv = StratifiedKFold(n_splits=self.cv_fold, shuffle=True)
            for train, test in cv.split(x_train, y_train):
                # fit the model
                model = clf.fit(x_train[train],y_train[train])
                # get predictions
                predict_label = model.predict(x_train[test])
                predict_proba = model.predict_proba(x_train[test])[:,1]
                # calculate metrics
                tot_acc += metrics.accuracy_score(y_train[test], predict_label)
                if score == "pr":
                    tot_score += metrics.average_precision_score(y_train[test], predict_proba)
                elif score == "auc" or score == "acc":
                    # if accuracy we still return auc
                    tot_score += metrics.roc_auc_score(y_train[test], predict_proba)
                else:
                    raise Exception("score metrics should be pr/auc/acc")
            # calculate average metrics
            acc = tot_acc/self.cv_fold
            scr = tot_score/self.cv_fold
            avg_acc += acc
            avg_scr += scr
        avg_acc = avg_acc/self.cv_num
        avg_scr = avg_scr/self.cv_num
        return {"params":comb_dict, "avg_acc": avg_acc, "avg_%s"%score: avg_scr}

    # TODO: separate the log
    def get_best_param(self, save_to_file=False, num_workers=os.cpu_count(), score="auc"):
        print("==== Get best param, utilizing %d workers =====" % num_workers)
        start_time = timeit.default_timer()
        comb_lst = list(self.param_grid.values())
        combinations = list(itertools.product(*comb_lst))

        # loop through every hyperparameter combination
        max_auc = 0
        run_kfold_partial = functools.partial(self.run_kfold, score=score)
        with cc.ProcessPoolExecutor(max_workers = num_workers) as executor:
            # TODO: update input to combinations to dictionary
            output = list(tqdm(executor.map(run_kfold_partial, combinations), total=len(combinations)))

            if save_to_file:
                # write preliminary output
                columns = ["n_est", "max_depth", "acc", "auc"]
                output_df = pd.DataFrame.from_records(output, columns = columns)
                output_df.to_csv("rf_all_params_results_step1.csv", index=False)

        best = max(output, key=lambda x:x['avg_%s'%score])

        print("Best params (using %s): %s" % (score,str(best["params"])))
        print("Total running time: %d seconds" % (timeit.default_timer() - start_time))
        return self.classifier(**best["params"])

    def get_topn(self, clf):
        model = clf.fit(self.x_train,self.y_train)
        feat_impt = model.feature_importances_
        print("Top %d features"%self.topn,sorted(zip(map(lambda x: round(x, 4), feat_impt), self.x_train.columns),
             reverse=True)[:self.topn])
        # get the new x
        self.x_train = self.train_data.iloc[:, feat_impt.argsort()[::-1][:self.topn]]
        df = self.x_train
        df['label'] = self.y_train
        return df
