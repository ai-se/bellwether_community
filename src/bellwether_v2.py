import pandas as pd
import numpy as np
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics

import SMOTE
import feature_selector
import DE
import CFS
import metrices
import measures

import sys
import traceback
import warnings
import os
import copy
import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
from typing import NoReturn
from collections import defaultdict

from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue

warnings.filterwarnings("ignore")

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class bellwether(object):

    def __init__(self,data_source1,data_source2):
        self.data_source1 = data_source1
        self.data_source2 = data_source2
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
            self.projects = list(self.projects1_list & self.projects2_list)

        self.cores = cpu_count()
    
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

    def bellwether(self,projects):
        final_score = {}
        for s_project in projects:
            print(s_project)
            try:
                df = pd.read_pickle(self.data_source1 + '/' + s_project + '_commit.pkl')
                df1 = df[df['buggy'] == 1]
                X1 = df1.commit_number
                X2 = df1.parent
                X = np.append(X1,X2)
                path = self.data_source2 + '/' + s_project + '.csv'
                df = self.prepare_data(path,X)
                df.reset_index(drop=True,inplace=True)
                y = df.fix
                X = df.drop(labels = ['fix'],axis = 1)
                kf = StratifiedKFold(n_splits = 5)
                goal = 'f1'
                learner = [SK_LR][0]
                F = {}
                #scores = {}
                score = {}
                for i in range(5):
                    for train_index, tune_index in kf.split(X, y):
                        X_train, X_tune = X.iloc[train_index], X.iloc[tune_index]
                        y_train, y_tune = y[train_index], y[tune_index]
                        _df = pd.concat([X_train,y_train], axis = 1)
                        _df_tune = pd.concat([X_tune,y_tune], axis = 1)
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
                                destination_df = pd.read_pickle(self.data_source1 + '/' + d_project + '_commit.pkl')
                                df1 = destination_df[destination_df['buggy'] == 1]
                                X1 = df1.commit_number
                                X2 = df1.parent
                                _X = np.append(X1,X2)
                                path = self.data_source2 + '/'+ d_project + '.csv'
                                destination_df = self.prepare_data(path,_X)
                                destination_df.reset_index(drop=True,inplace=True)
                                destination_df = destination_df[selected_cols]
                                test_y = destination_df.fix
                                test_X = destination_df.drop(labels = ['fix'],axis = 1)
                                clf = learner(X_train, y_train,  test_X,test_y, goal)
                                F = clf.learn(F,**params)
                                if d_project not in score.keys():
                                    score[d_project] = F
                                else:
                                    score[d_project]['f1'].append(F['f1'][0])
                                    score[d_project]['precision'].append(F['precision'][0])
                                    score[d_project]['recall'].append(F['recall'][0])
                                    score[d_project]['g-score'].append(F['g-score'][0])
                                    score[d_project]['d2h'].append(F['d2h'][0])

                            except:
                                print(s_project,d_project,sys.exc_info())
                                continue
                final_score[s_project] = score
            except:
                print(s_project,sys.exc_info())
                continue
        print(final_score)
        return final_score

    def run_bellwether(self):
        threads = []
        results = {}
        self.projects = self.projects[0:2]
        projects = np.array_split(self.projects, self.cores)
        for i in range(self.cores):
            print("starting thread ",i)
            t = ThreadWithReturnValue(target = self.bellwether, args = [projects[i]])
            threads.append(t)
        for th in threads:
            th.start()
        for th in threads:
            response = th.join()
            results.update(response)
        with open('data/bellwether.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)




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

    def apply_smote(self,df,neighbours,r):
        cols = df.columns
        smt = SMOTE.smote(df,neighbor = neighbours,r = r)
        df = smt.run()
        df.columns = cols
        return df

    def learn(self,F, **kwargs):
        """
        :param F: a dict, holds all scores, can be used during debugging
        :param kwargs: a dict, all the parameters need to set after tuning.
        :return: F, scores.
        """
        self.scores = {self.goal: [0.0]}
        try:
            neighbours = kwargs.pop('neighbours')
            r = kwargs.pop('r')
            self.learner.set_params(**kwargs)
            _df = pd.concat([self.train_X, self.train_Y], axis = 1)
            _df = self.apply_smote(_df,neighbours,r)
            y_train = _df.fix
            X_train = _df.drop(labels = ['fix'],axis = 1)
            predict_result = []
            clf = self.learner.fit(X_train, y_train)
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
        if self.goal in ['f1','precision','recall','g-score','d2h']:
            F['f1'] = [abcd.calculate_f1_score()]
            F['precision'] = [abcd.calculate_precision()]
            F['recall'] = [abcd.calculate_recall()]
            F['g-score'] = [abcd.get_g_score()]
            F['d2h'] = [abcd.calculate_d2h()]
            return F
        else:
            print("wronging goal")
            return F

    def predict(self,test_X):
        return self.learner.predict(test_X)



class SK_LR(DE_Learners):
    def __init__(self, train_x, train_y, predict_x, predict_y, goal):
        clf = LogisticRegression()
        super(SK_LR, self).__init__(clf, train_x, train_y, predict_x, predict_y,goal)

    def get_param(self):
        tunelst = {"penalty": ['l1', 'l2','elasticnet','none'],
                   "multi_class": ['ovr', 'multinomial','auto'],
                   "C": [1.0,200.0],
                   "dual": [True, False],
                   "fit_intercept": [True, False],
                   "intercept_scaling": [1.0,100.0],
                   "class_weight": ["balanced", 'none'],
                   "solver": ['newton-cg','lbfgs','liblinear','sag', 'saga'],
                   "warm_start": [True, False],
                   "max_iter": [100,600],
                   "neighbours": [5,21],
                   "r":[1,6]}
        return tunelst

if __name__ == "__main__":
    #path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data'
    path = '/gpfs_common/share02/tjmenzie/smajumd3/AI4SE/bellwether_community/data'
    bell = bellwether(path + '/data',
                                path + '/commit_guru')
    bell.run_bellwether()