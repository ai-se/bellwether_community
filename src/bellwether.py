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
from sklearn import svm
import sys
import os
import copy
import feature_selector
import warnings

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

warnings.filterwarnings("ignore")
class bellwether(object):

    def __init__(self,data_source,model):
        self.data_source = data_source
        self.model = model
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            _dir = data_source + '/'
        else:
            _dir = data_source + '\\'
        self.projects = [join(_dir, f) for f in listdir(_dir) if isfile(join(_dir, f))]

    def get_data(self,data):
        y = data['BUGS'].values
        X = data.drop(labels=['BUGS'],axis = 1)
        return X,y

    def run_bellwether(self):
        model = self.model_selector()
        for s_project in self.projects:
            source_df = pd.read_csv(s_project)
            train_X, train_y = self.get_data(source_df)
            #clf = model.fit(train_X,train_y) # Create the model seletor and initializer
            destination_projects = copy.deepcopy(self.projects)
            destination_projects.remove(s_project)
            for d_project in destination_projects:
                destination_df = pd.read_csv(d_project)
                test_X, test_y = self.get_data(destination_df)




    def model_selector(self):
        return 0