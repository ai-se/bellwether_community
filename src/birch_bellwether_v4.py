import pandas as pd
import numpy as np
import math
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import copy
import traceback
import timeit



import matplotlib.pyplot as plt

import SMOTE
import feature_selector
# import DE
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

    def __init__(self,data_path,attr_df):
        self.data_path = data_path
        self.attr_df = attr_df
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
        cluster,cluster_tree,max_depth = self.cluster_driver(self.attr_df)
        return cluster,cluster_tree,max_depth

    def get_clusters(self,data_source):
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            _dir = data_source + '/'
        else:
            _dir = data_source + '\\'

        clusters = [(join(_dir, f)) for f in listdir(_dir) if Path(join(_dir, f)).is_dir()]
        return clusters

    def norm(self,x,df):
        lo = df.min()
        hi = df.max()
        return (x - lo) / (hi - lo +0.00000001)

    def dominate(self,_df,t,row_project_name,goals):
        wins = 0
        for i in range(_df.shape[0]):
            project_name = _df.iloc[i].name
            row = _df.iloc[i].tolist()
            if project_name != row_project_name:
                if self.dominationCompare(row, t,goals,_df):
                    wins += 1
        return wins

    def dominationCompare(self,other_row, t,goals,df):
        n = len(goals)
        weight = {'recall':1,'precision':1,'pf':-1.5}
        sum1, sum2 = 0,0
        for i in range(len(goals)):
            _df = df[goals[i]]
            w = weight[goals[i]]
            x = t[i]
            y = other_row[i]
            x = self.norm(x,_df)
            y = self.norm(y,_df)
            sum1 = sum1 - math.e**(w * (x-y)/n)
            sum2 = sum2 - math.e**(w * (y-x)/n)
        return sum1/n < sum2/n


    def bellwether(self,selected_projects,all_projects):
        final_score = {}
        count = 0
        for s_project in selected_projects:
            try:
                print(s_project,selected_projects.shape[0])
                s_path = self.data_path + s_project
                #print(s_project)
                df = self.prepare_data(s_path)
                if df.shape[0] < 50:
                    continue
                else:
                    count+=1
                df.reset_index(drop=True,inplace=True)
                d = {'buggy': True, 'clean': False}
                df['Buggy'] = df['Buggy'].map(d)
                df, s_cols = self.apply_cfs(df)
                # s_cols = df.columns.tolist()
                df = self.apply_smote(df)
                y = df.Buggy
                X = df.drop(labels = ['Buggy'],axis = 1)
                kf = StratifiedKFold(n_splits = 5)
                score = {}
                F = {}
                for i in range(1):
                    clf = LogisticRegression()
                    # clf = RandomForestClassifier()
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

    def run_bellwether(self,projects):
        threads = []
        results = {}
        _projects = projects
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

    def run_bellwether_v2(self,_attr_df_train,cluster_ids,data_store_path):
        for cluster_id in cluster_ids:
            selected_projects = np.array(_attr_df_train.iloc[cluster_tree[cluster_id].data_points].index)
            self.run(selected_projects,cluster_id,data_store_path)

    def run(self,selected_projects,cluster_id,data_store_path):
        print(cluster_id)
        final_score = self.bellwether(selected_projects,selected_projects)
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
    
    def find_bellwether_level2(self,data_source,other_projects,path,fold):
        goals = ['recall','precision','pf']
        clusters = self.get_clusters(data_source)
        for cluster in clusters:
            if cluster.rsplit('/',1)[1] == 'results' or cluster.rsplit('/',1)[1] == 'cdom_level1':
                continue
            projects_performance = {}
            for goal in goals:
                df = pd.read_csv(cluster + '/1385_LR_bellwether_' + goal + '.csv')
                for row in range(df.shape[0]):
                    j = df.iloc[row].values[1:]
                    j_med = np.median(j)
                    project_name = df.iloc[row].values[0]
                    if project_name not in projects_performance.keys():
                        projects_performance[project_name] = {}
                    projects_performance[project_name][goal] = j_med
            _df = pd.DataFrame.from_dict(projects_performance, orient = 'index')
            dom_score = []
            for row_id in range(_df.shape[0]):
                project_name = _df.iloc[row_id].name
                row = _df.iloc[row_id].tolist()
                wins = self.dominate(_df,row,project_name,goals)
                dom_score.append(wins)
            _df['wins'] = dom_score
            _df.to_csv(cluster + '/cdom_latest.csv')
        return projects_performance

    def calculate_level_1_performance(self,data_source,clusters,path,fold,cluster,cluster_tree):
        df_train = pd.read_pickle(data_source + '/train_data.pkl')
        #cluster,cluster_tree = self.build_BIRCH(df_train)
        cluster_ids = []
        cluster_structure = {}
        size = {}
        for key in cluster_tree:
            if cluster_tree[key].depth != None:
                cluster_ids.append(key)
                if cluster_tree[key].depth not in cluster_structure.keys():
                    cluster_structure[cluster_tree[key].depth] = {}
                cluster_structure[cluster_tree[key].depth][key] = cluster_tree[key].parent_id
                size[key] = cluster_tree[key].size
        goals = ['recall','precision','pf','pci_20','ifa']
        count = 0
        score = []
        score_med = []
        cluster_info = {}
        for cluster in clusters:
            if cluster.rsplit('/',1)[1] in ['results','cdom_level1']:
                continue
            df = pd.read_csv(cluster + '/cdom_latest.csv')
            counts = {}
            med_count = []
            c_dom = df.wins.values.tolist()
            best_project = df.iloc[c_dom.index(max(c_dom)),0]
            for goal in goals:
                goal_df = pd.read_csv(cluster + '/1385_LR_bellwether_' + goal + '.csv')
                goal_df.rename(columns={'Unnamed: 0':'projects'},inplace=True)
                j = goal_df[goal_df['projects'] == best_project].values[0][1:]
                if goal == 'pci_20': # check number of projects >= 0.4 when goal is pci_20
                    value = sum(i >= 0.40 for i in j)
                elif goal != 'pf': # check number of projects >= 0.67 when goal is other then pci_20 and pf
                    value = sum(i >= 0.67 for i in j)
                else: # check number of projects <= 0.33 when goal is pf
                    value = sum(i <= 0.33 for i in j)
                counts[goal] = value
            score_med.append([int(cluster.rsplit('/',1)[1]),goal_df.shape[0],
                            counts['recall'],
                            counts['precision'],
                            counts['pf'],
                            counts['pci_20'],
                            max(c_dom),
                            best_project])
        score_df = pd.DataFrame(score_med, columns = ['id','Total_projects','count_recall',
                                                    'count_precision','count_pf','count_pci_20',
                                                    'cdom_score','bellwether'])
        score_df = score_df.sort_values('id')
        score_df.to_csv(data_source + '/bellwether_cdom_2.csv')
        level_1_bellwethers = {}
        for cluster in cluster_structure[2].keys():
            if cluster_structure[2][cluster] not in level_1_bellwethers.keys():
                level_1_bellwethers[cluster_structure[2][cluster]] = []
            level_1_bellwethers[cluster_structure[2][cluster]].append(score_df[score_df['id'] == cluster].bellwether.values[0])
        score_med = []
        for key in  level_1_bellwethers.keys():
            sub_cluster_bellwethers = level_1_bellwethers[key]
            #bell = birch_bellwether.bellwether(path,df_train)
            final_score = self.bellwether(sub_cluster_bellwethers,sub_cluster_bellwethers)
            with open(data_source + '/cdom_level1/cluster_'  + str(key) + '_performance.pkl', 'wb') as handle:
                pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL)  

    def find_bellwether_level1(self,data_source,clusters,path,fold,cluster,cluster_tree):
        df_train = pd.read_pickle(data_source + '/train_data.pkl')
        #cluster,cluster_tree = self.build_BIRCH(df_train)
        cluster_ids = []
        cluster_structure = {}
        size = {}
        for key in cluster_tree:
            if cluster_tree[key].depth != None:
                cluster_ids.append(key)
                if cluster_tree[key].depth not in cluster_structure.keys():
                    cluster_structure[cluster_tree[key].depth] = {}
                cluster_structure[cluster_tree[key].depth][key] = cluster_tree[key].parent_id
                size[key] = cluster_tree[key].size
        goals = ['recall','precision','pf']
        score_df = pd.read_csv(data_source + '/bellwether_cdom_2.csv')
        score_df.drop(labels = ['Unnamed: 0'], axis = 1 ,inplace = True)
        level_1_bellwethers = {}
        for cluster in cluster_structure[2].keys():
            if cluster_structure[2][cluster] not in level_1_bellwethers.keys():
                level_1_bellwethers[cluster_structure[2][cluster]] = []
            level_1_bellwethers[cluster_structure[2][cluster]].append(score_df[score_df['id'] == cluster].bellwether.values[0])
        for cluster in cluster_structure[1].keys():
            if cluster not in level_1_bellwethers.keys():
                level_1_bellwethers[cluster] = []
            level_1_bellwethers[cluster] = list(df_train.iloc[cluster_tree[cluster].data_points].index)
        bell_df = {}
        for key in  level_1_bellwethers.keys():
            sub_cluster_bellwethers = level_1_bellwethers[key]
            final_score = pd.read_pickle(data_source + '/cdom_level1/cluster_'  + str(key) + '_performance.pkl')
            _results = {}
            for goal in goals:    
                for s_project in final_score.keys():
                    if s_project not in _results.keys():
                        _results[s_project] = {}
                        _temp = []
                    for d_projects in final_score[s_project].keys():
                        if goal == 'g':
                            _goal = 'g-score'
                        else:
                            _goal = goal
                        _score = np.median(final_score[s_project][d_projects][_goal])
                        _temp.append(np.median(final_score[s_project][d_projects][_goal]))
                    if goal not in _results[s_project].keys():
                        _results[s_project][goal] = []
                    _results[s_project][goal] = np.median(_temp)
            _df = pd.DataFrame.from_dict(_results, orient = 'index')
            dom_score = []
            for row_id in range(_df.shape[0]):
                project_name = _df.iloc[row_id].name
                row = _df.iloc[row_id].tolist()
                wins = self.dominate(_df,row,project_name,goals)
                dom_score.append(wins)
            _df['wins'] = dom_score
            c_dom = _df.wins.values.tolist()
            best_project = _df.index[c_dom.index(max(c_dom))]
            best_project_perf = _df.loc[best_project].values.tolist()
            best_project_perf.append(best_project)
            bell_df[key] = best_project_perf
        perf_df = pd.DataFrame.from_dict(bell_df, orient = 'index', columns = ['recall','precision','pf','cdom','bellwether'])    
        perf_df.to_csv(data_source + '/bellwether_cdom_1.csv')  

    def find_bellwether_level0(self,data_source,path,fold,cluster,cluster_tree):
        df_train = pd.read_pickle(data_source + '/train_data.pkl')
        #cluster,cluster_tree = self.build_BIRCH(df_train)
        cluster_ids = []
        cluster_structure = {}
        size = {}
        for key in cluster_tree:
            if cluster_tree[key].depth != None:
                cluster_ids.append(key)
                if cluster_tree[key].depth not in cluster_structure.keys():
                    cluster_structure[cluster_tree[key].depth] = {}
                cluster_structure[cluster_tree[key].depth][key] = cluster_tree[key].parent_id
                size[key] = cluster_tree[key].size
        goals = ['recall','precision','pf']
        bell_df = {}
        score_df = pd.read_csv(data_source + '/bellwether_cdom_1.csv')
        score_df = score_df.rename(columns = {'Unnamed: 0':'id'})
        _cluster_bellwethers = score_df.bellwether.values.tolist()
        #bell = birch_bellwether.bellwether(path,score_df)
        final_score = self.bellwether(_cluster_bellwethers,_cluster_bellwethers)
        _results = {}
        for goal in goals:    
            for s_project in final_score.keys():
                if s_project not in _results.keys():
                    _results[s_project] = {}
                    _temp = []
                for d_projects in final_score[s_project].keys():
                    if goal == 'g':
                        _goal = 'g-score'
                    else:
                        _goal = goal
                    _score = np.median(final_score[s_project][d_projects][_goal])
                    _temp.append(np.median(final_score[s_project][d_projects][_goal]))
                if goal not in _results[s_project].keys():
                    _results[s_project][goal] = []
                _results[s_project][goal] = np.median(_temp)
        _df = pd.DataFrame.from_dict(_results, orient = 'index')
        dom_score = []
        for row_id in range(_df.shape[0]):
            project_name = _df.iloc[row_id].name
            row = _df.iloc[row_id].tolist()
            wins = self.dominate(_df,row,project_name,goals)
            dom_score.append(wins)
        _df['wins'] = dom_score
        print(_df)
        c_dom = _df.wins.values.tolist()
        best_project = _df.index[c_dom.index(max(c_dom))]
        best_project_perf = _df.loc[best_project].values.tolist()
        best_project_perf.append(best_project)
        bell_df[key] = best_project_perf
        perf_df = pd.DataFrame.from_dict(bell_df, orient = 'index', columns = ['recall','precision','pf','cdom','bellwether'])    
        perf_df.to_csv(data_source + '/bellwether_cdom_0.csv')


if __name__ == "__main__":
    start = timeit.default_timer()
    #path = '/gpfs_common/share02/tjmenzie/smajumd3/AI4SE/bellwether_community/data/1385/converted'
    path = '/Users/suvodeepmajumder/Documents/AI4SE/bellwether_comminity/data/1385/converted'
    meta_path = 'data/1385/projects/selected_attr.pkl'
    _data_store_path = 'data/1385/new_exp/100/level_2/'
    attr_dict = pd.read_pickle(meta_path)
    attr_df = pd.DataFrame.from_dict(attr_dict,orient='index')
    # attr_df = attr_df[0:400]
    # attr_df.reset_index(drop=True,inplace=True)
    attr_df_index = list(attr_df.index)
    kf = KFold(n_splits=10,random_state=24)
    i = 0
    for train_index, test_index in kf.split(attr_df):
        data_store_path = _data_store_path
        _train_index = []
        _test_index = []
        for index in train_index:
            _train_index.append(attr_df_index[index])
        for index in test_index:
            _test_index.append(attr_df_index[index])
        data_store_path = data_store_path + 'fold_' + str(i) + '/'
        i += 1
        _attr_df_train = attr_df.loc[_train_index]
        #_attr_df_train.reset_index(drop=True,inplace=True)
        _attr_df_test = attr_df.loc[_test_index]
        #_attr_df_test.reset_index(drop=True,inplace=True)
        data_path = Path(data_store_path)
        if not data_path.is_dir():
            os.makedirs(data_path)
        _attr_df_train.to_pickle(data_store_path + 'train_data.pkl')
        _attr_df_test.to_pickle(data_store_path + 'test_data.pkl')
        bell = bellwether(path,_attr_df_train)
        cluster,cluster_tree,max_depth = bell.build_BIRCH()
        # print(cluster_tree)
        #with open('data/1385/exp1/1385_cluster_0.pkl', 'rb') as handle:
        #    _cluster_projects = pickle.load(handle)
        cluster_ids = []
        for key in cluster_tree:
            if cluster_tree[key].depth == 0:
                cluster_ids.append(key)
        #cluster_ids = [0] # need to include cluster 1
        # print((_attr_df_train.iloc[cluster_tree[cluster_ids[0]].data_points]))
        split_cluster_ids = np.array_split(cluster_ids,bell.cores)
        threads = []
        # results = {}
        for i in range(bell.cores):
            print("starting thread ",i)
            ids = list(split_cluster_ids[i]) #_attr_df_train,cluster_ids,data_store_path
            t = ThreadWithReturnValue(target = bell.run_bellwether_v2, args = [_attr_df_train,ids,data_store_path])
            threads.append(t)
        for th in threads:
            th.start()
        for th in threads:
            response = th.join()
            # results.update(response)
        # for ids in cluster_ids:
        #     selected_projects = list(_attr_df_train.iloc[cluster_tree[ids].data_points].index)
        #     bell.run(selected_projects,ids,data_store_path)
        mid = timeit.default_timer()
        # _dir = path + '/'
        # projects = [f for f in listdir(_dir) if isfile(join(_dir, f))]
        # bell.find_bellwether_level2(data_store_path,projects,path,i)
        # clusters = [(join(_dir, f)) for f in listdir(_dir) if Path(join(_dir, f)).is_dir()]
        # bell.calculate_level_1_performance(data_store_path,clusters,path,i,cluster,cluster_tree)
        # bell.find_bellwether_level1(data_store_path,clusters,path,i,cluster,cluster_tree)
        # bell.find_bellwether_level0(data_store_path,path,i,cluster,cluster_tree)
        break

    stop = timeit.default_timer() 
    print("Model training time: ", mid - start)
    print("total time: ", stop - start)
