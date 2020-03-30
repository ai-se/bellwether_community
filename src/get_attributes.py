import pandas as pd
import numpy as np
import pickle

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

import SMOTE
import feature_selector
import DE
import CFS
import metrices
import measures

from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue


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

class attribute(object):

    def __init__(self,path):
        self.path = path
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            self._dir = self.path + '/'
        else:
            self._dir = self.path + '\\'
        self.projects = [f for f in listdir(self._dir) if isfile(join(self._dir, f))]
        #self.projects = self.projects[0:10]
        self.cores = cpu_count()

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
        _cols = df.columns
        y = df.Buggy.values
        X = df.drop(labels = ['Buggy'],axis = 1)
        X = X.values
        selected_cols = CFS.cfs(X,y)
        fss = []
        imp_fss = []
        cols = df.columns[[selected_cols]].tolist()
        cols.append('Buggy')
        for col in _cols:
            _pos = cols.index(col) if col in cols else 0
            if col in cols:
                fss.append(1)
                imp_fss.append(_pos)
            else:
                fss.append(0)
                imp_fss.append(0)
        return df[cols],cols,fss,imp_fss
        
    def apply_smote(self,df):
        cols = df.columns
        smt = SMOTE.smote(df)
        df = smt.run()
        df.columns = cols
        return df

    def get_attributes(self,projects):
        count = 0
        project_selection = {}
        imp_project_selection = {}
        for project in projects:
            try:
                project_attr = []
                imp_project_attr = []
                path = self._dir + project
                print(project)
                df = self.prepare_data(path)
                if df.shape[0] >= 50:
                    continue
                else:
                    count+=1
                for _ in range(10):
                    _,_,fss,imp_fss = self.apply_cfs(df)
                    project_attr.append(fss)
                    imp_project_attr.append(imp_fss)
                project_attr = np.array(list(map(sum,zip(*project_attr))))/len(project_attr)
                imp_project_attr = np.array(list(map(sum,zip(*imp_project_attr))))/len(imp_project_attr)
                project_attr = [round(x) for x in project_attr]
                imp_project_attr = [round(x) for x in imp_project_attr]
                project_selection[project] = project_attr   
                imp_project_selection[project] = imp_project_attr
            except Exception as e:
                print(e)
                continue
        return project_selection,imp_project_selection

    def run_attributes_selector(self):
        threads = []
        results = {}
        imp_results = {}
        split_projects = np.array_split(self.projects, self.cores)
        for i in range(self.cores):
            print("starting thread ",i)
            t = ThreadWithReturnValue(target = self.get_attributes, args = [split_projects[i]])
            threads.append(t)
        for th in threads:
            th.start()
        for th in threads:
            response1,response2 = th.join()
            results.update(response1)
            imp_results.update(response2)
        return results,imp_results


if __name__ == "__main__":
    path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/1385/converted'
    #path = '/gpfs_common/share02/tjmenzie/smajumd3/AI4SE/bellwether_community/data/1385/converted'
    attr = attribute(path)
    project_selection,imp_results = attr.run_attributes_selector()
    print(imp_results)
    with open('data/1385/projects/new_selected_attr.pkl', 'wb') as handle:
            pickle.dump(project_selection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/1385/projects/imp_selected_attr.pkl', 'wb') as handle:
            pickle.dump(imp_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
