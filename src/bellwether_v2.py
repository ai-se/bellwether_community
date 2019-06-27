import pandas as pd
import numpy as np
import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
from typing import NoReturn
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import svm
import sys
import os
import copy
import feature_selector
from sklearn import metrics
import pickle

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

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

warnings.filterwarnings("ignore")
class bellwether(object):

    def __init__(self,data_source1,data_source2,model):
        self.data_source1 = data_source1
        self.data_source2 = data_source2
        self.model = model
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            _dir1 = self.data_source1 + '/'
            _dir2 = self.data_source2 + '/'
        else:
            _dir1 = self.data_source1 + '\\'
            _dir2= self.data_source2 + '\\'    
        self.projects1 = [f for f in listdir(_dir1) if isfile(join(_dir1, f))]
        self.projects2 = [f for f in listdir(_dir2) if isfile(join(_dir2, f))]
        self.projects1_list = []
        for project in self.projects1:
            x = project.split('_commit')[0]
            if x not in self.projects1_list:
                self.projects1_list.append(x)
        self.projects1_list = set(self.projects1_list)

        self.projects2_list = []
        for project in self.projects2:
            x = project.split('.csv')[0]
            if x not in self.projects2_list:
                self.projects2_list.append(x)
        self.projects2_list = set(self.projects2_list)

        if (self.projects1_list & self.projects2_list): 
            self.projects = self.projects1_list & self.projects2_list
    
    def prepare_data(self,path,X):
        df = pd.read_csv(path)
        df = df[df['commit_hash'].isin(X)]
        df = df.drop(labels = ['commit_hash', 'author_name', 'author_date_unix_timestamp',
        'author_email', 'author_date', 'commit_message','classification', 'linked', 'contains_bug', 'fixes',
                        'fileschanged','glm_probability', 'rf_probability',
        'repository_id', 'issue_id', 'issue_date', 'issue_type'],axis=1)
        df = df.dropna()
        df = df[['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'ndev', 'age',
            'nuc', 'exp', 'rexp', 'sexp','fix']]
        df = df.astype(np.float64)
        return df

    def get_features(self,df):
        fs = feature_selector.featureSelector()
        df,_feature_nums,features = fs.cfs_bfs(df)
        return df,features

    def apply_cfs(self,df):
        y = df.fix.values
        X = df.drop(labels = ['fix'],axis = 1)
        X = X.values
        selected_cols = CFS.cfs(X,y)
        cols = df.columns[[selected_cols]].tolist()
        cols.append('fix')
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

    def run_bellwether(self):
        final_score = {}
        for s_project in self.projects:
            print(s_project)
            try:
                df = pd.read_pickle('/Users/suvodeepmajumder/Documents/AI4SE/git_miner/data/' + s_project + '_commit.pkl')
                df1 = df[df['buggy'] == 1]
                X1 = df1.commit_number
                X2 = df1.parent
                X = np.append(X1,X2)
                path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/commit_guru/' + s_project + '.csv'
                df = self.prepare_data(path,X)
                df.reset_index(drop=True,inplace=True)
                y = df.fix
                X = df.drop(labels = ['fix'],axis = 1)
                kf = StratifiedKFold(n_splits = 10)
                goal = 'f1'
                learner = [SK_LR][0]
                F = {}
                #scores = {}
                score = {}
                for i in range(1):
                    for train_index, tune_index in kf.split(X, y):
                        X_train, X_tune = X.iloc[train_index], X.iloc[tune_index]
                        y_train, y_tune = y[train_index], y[tune_index]
                        _df = pd.concat([X_train,y_train], axis = 1)
                        _df_tune = pd.concat([X_tune,y_tune], axis = 1)
                        _df = self.apply_smote(_df)
                        _df,selected_cols = self.apply_cfs(_df)
                        y_train = _df.fix
                        X_train = _df.drop(labels = ['fix'],axis = 1)
                        _df_tune = _df_tune[selected_cols]
                        y_tune = _df_tune.fix
                        X_tune = _df_tune.drop(labels = ['fix'],axis = 1)
                        params, evaluation = self.tune_learner(learner, X_train, y_train,  X_tune,y_tune, goal)
                        destination_projects = copy.deepcopy(self.projects)
                        destination_projects.remove(s_project)
                        for d_project in destination_projects:
                            try:
                                destination_df = pd.read_pickle('/Users/suvodeepmajumder/Documents/AI4SE/git_miner/data/' + d_project + '_commit.pkl')
                                df1 = destination_df[destination_df['buggy'] == 1]
                                X1 = df1.commit_number
                                X2 = df1.parent
                                _X = np.append(X1,X2)
                                path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/commit_guru/' + d_project + '.csv'
                                destination_df = self.prepare_data(path,_X)
                                destination_df.reset_index(drop=True,inplace=True)
                                destination_df = destination_df[selected_cols]
                                test_y = destination_df.fix
                                test_X = destination_df.drop(labels = ['fix'],axis = 1)
                                clf = learner(X_train, y_train,  test_X,test_y, goal)
                                F = clf.learn(F,**params)
                                if d_project not in score.keys():
                                    score[d_project] = [F[goal][0]]
                                else:
                                    score[d_project].append(F[goal][0])
                            except:
                                print(s_project,d_project,sys.exc_info())
                                continue
                final_score[s_project] = score
            except:
                print(s_project,sys.exc_info())
                continue
        #df = pd.DataFrame(final_score)
        with open('data/bellwether.pkl', 'wb') as handle:
            pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #final_score.to_pickle('data/bellwether.pkl')

    def model_selector(self):
        #clf = DecisionTreeClassifier(criterion='entropy')
        #clf = SVC()
        #clf = ExtraTreesClassifier()
        clf = LogisticRegression(penalty='l1')
        return clf



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