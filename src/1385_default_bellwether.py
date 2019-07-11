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
warnings.filterwarnings("ignore")

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
    def __init__(self,data_source):
        self.data_source = data_source
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            _dir1 = self.data_source + '/'
        else:
            _dir1 = self.data_source + '\\' 
        self.projects = [f for f in listdir(_dir1) if isfile(join(_dir1, f))]
        self.selected_projects = self.get_eligible_projects()
        self.selected_projects = self.selected_projects[0:20]
        self.cores = cpu_count()

    
    def get_eligible_projects(self):
        selected_projects = []
        unselected_projects = []
        for project in self.projects:
            try:
                path = self.data_source + '/' + project
                df = self.prepare_data(path)
                if df.shape[0] >= 50:
                    selected_projects.append(project)
                else:
                    unselected_projects.append(project)
            except:
                continue
        return selected_projects
    
    def prepare_data(self,path):
        df = pd.read_csv(path)
        df = df.drop(labels = ['Host','Vcs','Project','File','PL','IssueTracking'],axis=1)
        df = df.dropna()
        df = df[['TLOC', 'TNF', 'TNC', 'TND', 'LOC', 'CL', 'NStmt', 'NFunc',
        'RCC', 'MNL', 'avg_WMC', 'max_WMC', 'total_WMC', 'avg_DIT', 'max_DIT',
        'total_DIT', 'avg_RFC', 'max_RFC', 'total_RFC', 'avg_NOC', 'max_NOC',
        'total_NOC', 'avg_CBO', 'max_CBO', 'total_CBO', 'avg_DIT.1',
        'max_DIT.1', 'total_DIT.1', 'avg_NIV', 'max_NIV', 'total_NIV',
        'avg_NIM', 'max_NIM', 'total_NIM', 'avg_NOM', 'max_NOM', 'total_NOM',
        'avg_NPBM', 'max_NPBM', 'total_NPBM', 'avg_NPM', 'max_NPM', 'total_NPM',
        'avg_NPRM', 'max_NPRM', 'total_NPRM', 'avg_CC', 'max_CC', 'total_CC',
        'avg_FANIN', 'max_FANIN', 'total_FANIN', 'avg_FANOUT', 'max_FANOUT',
        'total_FANOUT', 'NRev', 'NFix', 'avg_AddedLOC', 'max_AddedLOC',
        'total_AddedLOC', 'avg_DeletedLOC', 'max_DeletedLOC',
        'total_DeletedLOC', 'avg_ModifiedLOC', 'max_ModifiedLOC',
        'total_ModifiedLOC','Buggy']]
        return df

    def get_features(self,df):
        fs = feature_selector.featureSelector()
        df,_feature_nums,features = fs.cfs_bfs(df)
        return df,features

    def apply_cfs(self,df):
        y = df.Buggy.values
        X = df.drop(labels = ['Buggy'],axis = 1)
        X = X.values
        selected_cols = CFS.cfs(X,y)
        cols = df.columns[[selected_cols]].tolist()
        cols.append('Buggy')
        return df[cols],cols
        
    def apply_smote(self,df):
        cols = df.columns
        smt = SMOTE.smote(df)
        df = smt.run()
        df.columns = cols
        return df


    def bellwether(self):
        final_score = {}
        count = 0
        for s_project in self.selected_projects:
            try:
                s_path = self.data_source + '/' + s_project
                print(s_project)
                df = self.prepare_data(s_path)
                if df.shape[0] < 50:
                    continue
                else:
                    count+=1
                df.reset_index(drop=True,inplace=True)
                d = {'buggy': True, 'clean': False}
                df['Buggy'] = df['Buggy'].map(d)
                y = df.Buggy
                X = df.drop(labels = ['Buggy'],axis = 1)
                kf = StratifiedKFold(n_splits = 5)
                score = {}
                F = {}
                for i in range(5):
                    for train_index, tune_index in kf.split(X, y):
                        X_train, X_tune = X.iloc[train_index], X.iloc[tune_index]
                        y_train, y_tune = y[train_index], y[tune_index]
                        clf = LogisticRegression()
                        clf.fit(X_train,y_train)
                        destination_projects = copy.deepcopy(self.selected_projects)
                        destination_projects.remove(s_project)
                        for d_project in destination_projects:
                            #print(d_project)
                            try:
                                d_path = self.data_source + '/' + d_project
                                dest_df = self.prepare_data(d_path)
                                if dest_df.shape[0] < 50:
                                    continue
                                dest_df.reset_index(drop=True,inplace=True)
                                d = {'buggy': True, 'clean': False}
                                dest_df['Buggy'] = dest_df['Buggy'].map(d)
                                test_y = dest_df.Buggy
                                test_X = dest_df.drop(labels = ['Buggy'],axis = 1)
                                predicted = clf.predict(test_X)
                                _df_test_loc = test_X.LOC
                                abcd = metrices.measures(test_y,predicted,_df_test_loc)
                                F['f1'] = [abcd.calculate_f1_score()]
                                F['precision'] = [abcd.calculate_precision()]
                                F['recall'] = [abcd.calculate_recall()]
                                F['g-score'] = [abcd.get_g_score()]
                                F['d2h'] = [abcd.calculate_d2h()]
                                F['pci_20'] = [abcd.get_pci_20()]
                                F['ifa'] = [abcd.get_ifa()]
                                F['pd'] = [abcd.get_pd()]
                                F['pf'] = [abcd.get_pf()]
                                _F = copy.deepcopy(F)
                                if 'f1' not in score.keys():
                                    score[d_project] = _F
                                else:
                                    score[d_project]['f1'].append(F['f1'][0])
                                    score[d_project]['precision'].append(F['precision'][0])
                                    score[d_project]['recall'].append(F['recall'][0])
                                    score[d_project]['g-score'].append(F['g-score'][0])
                                    score[d_project]['d2h'].append(F['d2h'][0])
                                    score[d_project]['pci_20'].append(F['pci_20'][0])
                                    score[d_project]['ifa'].append(F['ifa'][0])
                                    score[d_project]['pd'].append(F['pd'][0])
                                    score[d_project]['pf'].append(F['pf'][0])
                            except Exception as e:
                                print(e)
                                continue
                    final_score[s_project] = score 
            except Exception as e:
                print(e)
                continue
        return final_score

    
    def run_bellwether(self):
        final_score = self.bellwether()
        with open('data/1385/20/1385_LR_default_bellwether_20.pkl', 'wb') as handle:
            pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
        df = pd.read_pickle('data/1385/20/1385_LR_default_bellwether_20.pkl')
        results_f1 = {}
        results_precision = {}
        results_recall = {}
        results_g = {}
        results_d2h = {}
        results_pci_20 = {}
        results_ifa = {}
        results_pd = {}
        results_pf = {}
        for s_project in df.keys():
            if s_project not in results_f1.keys():
                results_f1[s_project] = {}
                results_precision[s_project] = {}
                results_recall[s_project] = {}
                results_g[s_project] = {}
                results_d2h[s_project] = {}
                results_pci_20[s_project] = {}
                results_ifa[s_project] = {}
                results_pd[s_project] = {}
                results_pf[s_project] = {}
            for d_projects in df[s_project].keys():
                results_f1[s_project][d_projects] = np.median(df[s_project][d_projects]['f1'])
                results_precision[s_project][d_projects] = np.median(df[s_project][d_projects]['precision'])
                results_recall[s_project][d_projects] = np.median(df[s_project][d_projects]['recall'])
                results_g[s_project][d_projects] = np.median(df[s_project][d_projects]['g-score'])
                results_d2h[s_project][d_projects] = np.median(df[s_project][d_projects]['d2h'])
                results_pci_20[s_project][d_projects] = np.median(df[s_project][d_projects]['pci_20'])
                results_ifa[s_project][d_projects] = np.median(df[s_project][d_projects]['ifa'])
                results_pd[s_project][d_projects] = np.median(df[s_project][d_projects]['pd'])
                results_pf[s_project][d_projects] = np.median(df[s_project][d_projects]['pf'])

        results_f1_df = pd.DataFrame.from_dict(results_f1, orient='index')
        results_precision_df = pd.DataFrame.from_dict(results_precision, orient='index')
        results_recall_df = pd.DataFrame.from_dict(results_recall, orient='index')
        results_g_df = pd.DataFrame.from_dict(results_g, orient='index')
        results_d2h_df = pd.DataFrame.from_dict(results_d2h, orient='index')
        results_pci_20_df = pd.DataFrame.from_dict(results_pci_20, orient='index')
        results_ifa_df = pd.DataFrame.from_dict(results_ifa, orient='index')
        results_pd_df = pd.DataFrame.from_dict(results_pd, orient='index')
        results_pf_df = pd.DataFrame.from_dict(results_pf, orient='index')

        results_f1_df.to_csv('data/1385/20/1385_LR_bellwether_f1.csv')
        results_precision_df.to_csv('data/1385/20/1385_LR_bellwether_precision.csv')
        results_recall_df.to_csv('data/1385/20/1385_LR_bellwether_recall.csv')
        results_g_df.to_csv('data/1385/20/1385_LR_bellwether_g.csv')
        results_d2h_df.to_csv('data/1385/20/1385_LR_bellwether_d2h.csv')
        results_pci_20_df.to_csv('data/1385/20/1385_LR_bellwether_pci_20.csv')
        results_ifa_df.to_csv('data/1385/20/1385_LR_bellwether_ifa.csv')
        results_pd_df.to_csv('data/1385/20/1385_LR_bellwether_pd.csv')
        results_pf_df.to_csv('data/1385/20/1385_LR_bellwether_pf.csv')





if __name__ == "__main__":
    #path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/1385/converted'
    path = '/gpfs_common/share02/tjmenzie/smajumd3/AI4SE/bellwether_community/data/1385/converted'
    bell = bellwether(path)
    bell.run_bellwether()
