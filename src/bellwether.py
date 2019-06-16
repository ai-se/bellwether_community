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
from sklearn.svm import SVC
from sklearn import svm
import sys
import os
import copy
import feature_selector
from sklearn import metrics
import warnings
import SMOTE

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

    def get_data(self,df):
        df = df.drop(labels = ['commit_hash', 'author_name', 'author_date_unix_timestamp',
            'author_email', 'author_date', 'commit_message','classification', 'linked', 'contains_bug', 'fixes',
                            'fileschanged','glm_probability', 'rf_probability',
            'repository_id', 'issue_id', 'issue_date', 'issue_type'],axis=1)
        df = df.dropna()
        df = df[['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'ndev', 'age',
            'nuc', 'exp', 'rexp', 'sexp','fix']]
        smt = SMOTE.smote(df)
        df = smt.run()
        df.columns = ['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'ndev', 'age',
            'nuc', 'exp', 'rexp', 'sexp','fix']
        y = df.fix
        y=y.astype('bool')
        X = df.drop(labels=['fix'],axis=1)
        #y = data['BUGS'].values
        #X = data.drop(labels=['BUGS'],axis = 1)
        return X,y

    def get_data1(self,df):
        df = df.drop(labels = ['commit_hash', 'author_name', 'author_date_unix_timestamp',
            'author_email', 'author_date', 'commit_message','classification', 'linked', 'contains_bug', 'fixes',
                            'fileschanged','glm_probability', 'rf_probability',
            'repository_id', 'issue_id', 'issue_date', 'issue_type'],axis=1)
        df = df.dropna()
        df = df[['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'ndev', 'age',
            'nuc', 'exp', 'rexp', 'sexp','fix']]
        y = df.fix
        y=y.astype('bool')
        X = df.drop(labels=['fix'],axis=1)
        #y = data['BUGS'].values
        #X = data.drop(labels=['BUGS'],axis = 1)
        return X,y

    def get_baseline(self):
        model = self.model_selector()
        final_score = []
        for s_project in self.projects:
            try:
                source_df = pd.read_csv(s_project)
                X, y = self.get_data1(source_df)
                train_X,test_X,train_y,test_y = train_test_split(X, y, test_size=0.33, random_state=42)
                clf = model.fit(train_X,train_y)
                predicted = clf.predict(test_X)
                fpr, tpr, thresholds = metrics.roc_curve(test_y, predicted, pos_label=True)
                score = metrics.auc(fpr, tpr)
                print(score)
                final_score.append(score)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                continue
        df = pd.DataFrame(final_score)
        df.to_csv('data/baseline.csv')

    def run_bellwether(self):
        model = self.model_selector()
        final_score = []
        for s_project in self.projects:
            try:
                print(s_project)
                project_score = []
                #project_score.insert(0,s_project)
                source_df = pd.read_csv(s_project)
                train_X, train_y = self.get_data(source_df)
                clf = model.fit(train_X,train_y) # Create the model seletor and initializer
                destination_projects = copy.deepcopy(self.projects)
                #destination_projects.remove(s_project)
                for d_project in destination_projects:
                    try:
                        destination_df = pd.read_csv(d_project)
                        test_X, test_y = self.get_data1(destination_df)
                        predicted = clf.predict(test_X)
                        score = metrics.roc_auc_score(test_y,predicted,average='weighted')
                        project_score.append(score)
                    except:
                        print(s_project,d_project)
                        continue
                print(project_score)
                final_score.append(project_score)
            except:
                print(s_project)
                continue
        df = pd.DataFrame(final_score)
        df.to_csv('data/bellwether.csv')



    def model_selector(self):
        #clf = DecisionTreeClassifier(criterion='entropy')
        #clf = SVC()
        #clf = ExtraTreesClassifier()
        clf = LogisticRegression(penalty='l1')
        return clf