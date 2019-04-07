import pandas as pd
import numpy as np
import feature_selector

class features(object):

    def __init__(self,data,fs):
        self.data = data
        self.fs = fs
        self.fec = feature_selector.featureSelector()

    def get_features(self):
        if self.fs == 'gain_rank':
            _,features = self.fec.gain_rank(self.data)
        elif self.fs == 'relief':
            _,features = self.fec.relief(self.data)
        elif self.fs == 'consistency_subset':
            _,features = self.fec.consistency_subset(self.data)
        elif self.fs == 'cfs':
            _,features = self.fec.cfs(self.data)
        elif self.fs == 'tfs':
            _,features = self.fec.tfs(self.data)
        elif self.fs == 'l1':
            _,features = self.fec.l1(self.data)
        else:
            print("Please select one of the existing feature selctor")
        return features
