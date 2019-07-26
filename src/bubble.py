import pandas as pd
import numpy as np
import math
import pickle

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
import copy
import traceback



import matplotlib.pyplot as plt

import SMOTE
import feature_selector
import DE
import CFS
import birch
import metrics.abcd


from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue

import metrices
import measures

import sys
import traceback
import warnings
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

    def __init__(self,data_path,meta_path):
        self.data_path = data_path
        self.meta_path = meta_path
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            self.data_path = self.data_path + '/'
        else:
            self.data_path = self.data_path + '\\'
        self.projects = [f for f in listdir(self.data_path) if isfile(join(self.data_path, f))]
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

    def load_data(self,path,target):
        df = pd.read_csv(path)
        if path == 'data/jm1.csv':
            df = df[~df.uniq_Op.str.contains("\?")]
        y = df[target]
        X = df.drop(labels = target, axis = 1)
        X = X.apply(pd.to_numeric)
        return X,y

    # Cluster Driver
    def cluster_driver(self,df,print_tree = True):
        X = df.apply(pd.to_numeric)
        cluster = birch.birch(branching_factor=20)
        #X.set_index('Project Name',inplace=True)
        cluster.fit(X)
        cluster_tree,max_depth = cluster.get_cluster_tree()
        #cluster_tree = cluster.model_adder(cluster_tree)
        if print_tree:
            cluster.show_clutser_tree()
        return cluster,cluster_tree,max_depth

    def build_BIRCH(self):
        attr_dict = pd.read_pickle(self.meta_path)
        self.attr_df = pd.DataFrame.from_dict(attr_dict,orient='index')
        cluster,cluster_tree,_ = self.cluster_driver(self.attr_df)
        return cluster,cluster_tree


    def bellwether(self,selected_projects,all_projects):
        final_score = {}
        count = 0
        for s_project in selected_projects:
            try:
                s_path = self.data_path + s_project
                print(s_project)
                df = self.prepare_data(s_path)
                if df.shape[0] < 50:
                    continue
                else:
                    count+=1
                df.reset_index(drop=True,inplace=True)
                d = {'buggy': True, 'clean': False}
                df['Buggy'] = df['Buggy'].map(d)
                df, s_cols = self.apply_cfs(df)
                df = self.apply_smote(df)
                y = df.Buggy
                X = df.drop(labels = ['Buggy'],axis = 1)
                kf = StratifiedKFold(n_splits = 5)
                score = {}
                F = {}
                for i in range(1):
                    #for train_index, tune_index in kf.split(X, y):
                    #X_train, X_tune = X.iloc[train_index], X.iloc[tune_index]
                    #y_train, y_tune = y[train_index], y[tune_index]
                    clf = LogisticRegression()
                    clf.fit(X,y)
                    destination_projects = copy.deepcopy(all_projects)
                        #destination_projects.remove(s_project)
                    for d_project in destination_projects:
                        try:
                            d_path = self.data_path + d_project
                            _test_df = self.prepare_data(d_path)
                            _df_test_loc = _test_df.LOC
                            test_df = _test_df[s_cols]
                            if test_df.shape[0] < 50:
                                continue
                            test_df.reset_index(drop=True,inplace=True)
                            d = {'buggy': True, 'clean': False}
                            test_df['Buggy'] = test_df['Buggy'].map(d)
                            test_y = test_df.Buggy
                            test_X = test_df.drop(labels = ['Buggy'],axis = 1)
                            predicted = clf.predict(test_X)
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
                            print("dest",d_project,e)
                            continue
                    final_score[s_project] = score 
            except Exception as e:
                print("src",s_project,e)
                continue
        return final_score

    def run_bellwether(self,projects,bellwethers):
        threads = []
        results = {}
        _projects = bellwethers
        split_projects = np.array_split(_projects, self.cores)
        for i in range(self.cores):
            print("starting thread ",i)
            t = ThreadWithReturnValue(target = self.bellwether, args = [split_projects[i],projects])
            threads.append(t)
        for th in threads:
            th.start()
        for th in threads:
            response = th.join()
            results.update(response)
        return results

    def run(self,selected_projects,cluster_id,data_store_path,bellwethers):
        print(cluster_id)
        final_score = self.run_bellwether(selected_projects,bellwethers)
        data_path = Path(data_store_path + str(cluster_id))
        if not data_path.is_dir():
            os.makedirs(data_path)
        with open(data_store_path + str(cluster_id)  + '/1385_LR_default_bellwether.pkl', 'wb') as handle:
            pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
        df = pd.read_pickle(data_store_path + str(cluster_id)  + '/1385_LR_default_bellwether.pkl')
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
        
        results_f1_df.to_csv(data_store_path + str(cluster_id)  + '/1385_LR_bellwether_f1.csv')
        results_precision_df.to_csv(data_store_path + str(cluster_id)  + '/1385_LR_bellwether_precision.csv')
        results_recall_df.to_csv(data_store_path + str(cluster_id)  + '/1385_LR_bellwether_recall.csv')
        results_g_df.to_csv(data_store_path + str(cluster_id)  + '/1385_LR_bellwether_g.csv')
        results_d2h_df.to_csv(data_store_path + str(cluster_id)  + '/1385_LR_bellwether_d2h.csv')
        results_pci_20_df.to_csv(data_store_path + str(cluster_id)  + '/1385_LR_bellwether_pci_20.csv')
        results_ifa_df.to_csv(data_store_path + str(cluster_id)  + '/1385_LR_bellwether_ifa.csv')
        results_pd_df.to_csv(data_store_path+ str(cluster_id)  + '/1385_LR_bellwether_pd.csv')
        results_pf_df.to_csv(data_store_path + str(cluster_id)  + '/1385_LR_bellwether_pf.csv')


if __name__ == "__main__":
    #path = '/gpfs_common/share02/tjmenzie/smajumd3/AI4SE/bellwether_community/data/1385/converted'
    path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/1385/converted'
    meta_path = 'data/1385/projects/selected_attr.pkl'
    data_store_path = 'data/1385/exp1/bubble/bogus'
    bell = bellwether(path,meta_path)
    cluster,cluster_tree = bell.build_BIRCH()
    bellwethers = ['benojt.csv','jmule.csv','botsnscouts.csv',
               'libarchive.csv','grisbi.csv','magicwars.csv',
               'turbotrader-bos.csv','xmemcached.csv','columba.csv',
               'enlightenment.csv','sqlpower-library.csv','hydrogen.csv',
               'nh3d.csv','mp-rechnungs-und-kundenverwaltung.csv','h2database.csv',
               'reaper-ecad.csv','alembik.csv','personalaccess.csv','emftriple.csv',
               'jchassis.csv','empyrean.csv','fbmanager.csv','growl-for-windows.csv',
               'quantlib.csv','javagroups.csv','twostep.csv','nativeclient.csv',
               'rebecca-aiml.csv','turbotrader-bos.csv','gpsmid.csv']
    with open('data/1385/exp1/1385_cluster_0.pkl', 'rb') as handle:
        _cluster_projects = pickle.load(handle)
    print(len(_cluster_projects))
    cluster_ids = [0] # need to include cluster 1
    for ids in cluster_ids:
        selected_projects = _cluster_projects
        bell.run(selected_projects,ids,data_store_path,bellwethers)
