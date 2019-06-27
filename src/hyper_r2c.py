import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import pickle



import matplotlib.pyplot as plt

import SMOTE
import feature_selector
import DE
import CFS

import metrices
import measures

import sys
import traceback
import warnings
warnings.filterwarnings("ignore")



class hyper(object):

    def __init__(self,path):
        self.path = path
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            _dir = self.path + '/'
        else:
            _dir = self.path + '\\'
        self.projects = [join(_dir, f) for f in listdir(_dir) if isfile(join(_dir, f))]

    def prepare_data(self,path):
        df = pd.read_csv(path)
        df = df.dropna()
        df = df.astype(np.float64)
        return df

    def get_features(self,df):
        fs = feature_selector.featureSelector()
        df,_feature_nums,features = fs.cfs_bfs(df)
        return df,features

    def apply_cfs(self,df):
        y = df.BUGS.values
        X = df.drop(labels = ['BUGS'],axis = 1)
        X = X.values
        selected_cols = CFS.cfs(X,y)
        cols = df.columns[[selected_cols]].tolist()
        cols.append('BUGS')
        return df[cols],cols
        
    def apply_smote(self,df):
        cols = df.columns
        smt = SMOTE.smote(df)
        df = smt.run()
        df.columns = cols
        return df

    def tune_learner(self,learner, train_X, train_Y, tune_X, tune_Y, goal,target_class=None):
        if not target_class:
            target_class = goal
        clf = learner(train_X, train_Y, tune_X, tune_Y, goal)
        tuner = DE.DE_Tune_ML(clf, clf.get_param(), goal, target_class)
        return tuner.Tune()

    def tune(self):
        project_scores = {}
        for project in self.projects:
            df = self.prepare_data(project)
            df.reset_index(drop=True,inplace=True)
            y = df.BUGS
            X = df.drop(labels = ['BUGS'],axis = 1)
            train_X,test_X,train_y,test_y = train_test_split(X, y, test_size=0.33, random_state=13)
            df_test = pd.concat([test_X,test_y], axis = 1)
            df = pd.concat([train_X,train_y], axis = 1)
            df.reset_index(drop=True,inplace=True)
            y = df.BUGS
            X = df.drop(labels = ['BUGS'],axis = 1)
            kf = StratifiedKFold(n_splits = 5)
            goal = 'f1'
            learner = [SK_LR][0]
            F = {}
            score = []
            for i in range(2):
                for train_index, tune_index in kf.split(X, y):
                    X_train, X_tune = X.iloc[train_index], X.iloc[tune_index]
                    y_train, y_tune = y[train_index], y[tune_index]
                    _df = pd.concat([X_train,y_train], axis = 1)
                    _df_tune = pd.concat([X_tune,y_tune], axis = 1)
                    _df = self.apply_smote(_df)
                    _df,selected_cols = self.apply_cfs(_df)
                    y_train = _df.BUGS
                    X_train = _df.drop(labels = ['BUGS'],axis = 1)
                    _df_tune = _df_tune[selected_cols]
                    y_tune = _df_tune.BUGS
                    X_tune = _df_tune.drop(labels = ['BUGS'],axis = 1)
                    _df_test = df_test[selected_cols]
                    test_y = _df_test.BUGS
                    test_X = _df_test.drop(labels = ['BUGS'],axis = 1)
                    params, evaluation = self.tune_learner(learner, X_train, y_train,  X_tune,y_tune, goal)
                    clf = learner(X_train, y_train,  test_X,test_y, goal)
                    F = clf.learn(F,**params)
                    score.append(F[goal][0])
            project_name = project.rsplit('/',1)[1].split('.',1)[0]
            project_scores[project_name] = score
            with open('data/r2c_hyper.pkl', 'wb') as handle:
                pickle.dump(project_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)



class DE_Learners(object):
    def __init__(self, clf, train_X, train_Y, test_X, test_Y, goal):
        """

        :param clf: classifier, SVM, etc...
        :param train_X: training data, independent variables.
        :param train_Y: training labels, dependent variables.
        :param predict_X: testing data, independent variables.
        :param predict_Y: testingd labels, dependent variables.
        :param goal: the objective of your tuning, F, recision,....
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.goal = goal
        self.param_distribution = self.get_param()
        self.learner = clf
        self.confusion = None
        self.params = None

    def learn(self,F, **kwargs):
        """
        :param F: a dict, holds all scores, can be used during debugging
        :param kwargs: a dict, all the parameters need to set after tuning.
        :return: F, scores.
        """
        self.scores = {self.goal: [0.0]}
        try:    
            self.learner.set_params(**kwargs)
            predict_result = []
            clf = self.learner.fit(self.train_X, self.train_Y)
            predict_result = clf.predict(self.test_X)
            self.abcd = metrices.measures(self.test_Y,predict_result)
            self.scores = self._Abcd(self.abcd,F)
            self.confusion = metrics.classification_report(self.test_Y.values.tolist(), predict_result, digits=2)
            self.params = kwargs
        except Exception as e:
            a = 10
        return self.scores
    
    def _Abcd(self,abcd , F):
        """

        :param predicted: predicted results(labels)
        :param actual: actual results(labels)
        :param F: previously got scores
        :return: updated scores.
        """
        if 'g-score' in self.goal:
            F['g-score'] = [abcd.get_g_score()]
            return F
        elif 'precision' in self.goal:
            F['precision'] = [abcd.get_precision()]
            return F
        elif 'f1' in self.goal:
            F['f1'] = [abcd.calculate_f1_score()]
            return F
        elif 'd2h' in self.goal:
            F['d2h'] = [abcd.calculate_d2h()]
            return F

    def predict(self,test_X):
        return self.learner.predict(test_X)

class SK_LR(DE_Learners):
    def __init__(self, train_x, train_y, predict_x, predict_y, goal):
        clf = LogisticRegression()
        super(SK_LR, self).__init__(clf, train_x, train_y, predict_x, predict_y,goal)

    def get_param(self):
        tunelst = {"penalty": ['l1', 'l2','elasticnet',None],
                   "multi_class": ['ovr', 'multinomial','auto'],
                   "C": [1.0,200.0],
                   "dual": [True, False],
                   "fit_intercept": [True, False],
                   "intercept_scaling": [1.0,100.0],
                   "class_weight": ["balanced", None],
                   "solver": ['newton-cg','lbfgs','liblinear','sag', 'saga'],
                   "warm_start": [True, False],
                   "max_iter": [100,600]}
        return tunelst



if __name__ == "__main__":
    data_path = '/gpfs_common/share02/tjmenzie/smajumd3/AI4SE/bellwether_community/src/datasets'
    #data_path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/src/datasets'
    hype = hyper(data_path)
    hype.tune()