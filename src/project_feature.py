import pandas as pd
import numpy as np
import feature_selector
import platform
from os.path import isfile, join
from glob import glob
from pathlib import Path
from typing import NoReturn
from os import listdir

class features(object):

    def __init__(self,data_source,fs,target):
        self.data_source = data_source
        self.target = target
        self.fs = fs
        self.fec = feature_selector.featureSelector()

    def get_projects(self):
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            _dir = self.data_source + '/'
        else:
            _dir = self.data_source + '\\'
        self.projects = [join(_dir, f) for f in listdir(_dir) if isfile(join(_dir, f))]

    def get_features(self,data):
        if self.fs == 'gain_rank':
            _,features = self.fec.gain_rank(data)
        elif self.fs == 'relief':
            _,features = self.fec.relief(data)
        elif self.fs == 'consistency_subset':
            _,features = self.fec.consistency_subset(data)
        elif self.fs == 'cfs':
            _,features = self.fec.cfs(data)
        elif self.fs == 'tfs':
            _,features = self.fec.tfs(data)
        elif self.fs == 'l1':
            _,features = self.fec.l1(data)
        else:
            print("Please select one of the existing feature selctor")
        return features


    def get_project_features(self):
        selected_features = []
        for s_project in self.projects:
            try: 
                index = s_project.split(self.data_source + '/')[1]
                source_df = pd.read_csv(s_project)
                self.columns_names = list(source_df.columns)
                _feature = list(self.get_features(source_df))
                _feature.insert(0,index)
                selected_features.append(_feature)
            except:
                print(s_project)
                continue
        self.columns_names = self.columns_names[0:len(self.columns_names)-1]
        self.columns_names.insert(0,'Project Name')
        selected_features_df = pd.DataFrame(selected_features,columns = self.columns_names)
        selected_features_df.set_index('Project Name',inplace=True)
        selected_features_df.to_csv('data/project_features.csv')
        return selected_features_df