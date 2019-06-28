import copy
import math

from sklearn import metrics
import numpy as np

class measures(object):

    def __init__(self,actual,predicted,labels = [0,1]):
        self.actual = actual
        self.predicted = predicted
        self.tn, self.fp, self.fn, self.tp = metrics.confusion_matrix(
            actual, predicted, labels=labels).ravel()
        self.pre, self.rec, self.spec, self.fpr, self.npv, self.acc, self.f1,self.pd,self.pf = self.get_performance()
        
    def calculate_recall(self):
        return round(metrics.recall_score(self.actual, self.predicted, average='weighted'),2)

    def calculate_precision(self):
        return round(metrics.precision_score(self.actual, self.predicted, average='weighted'),2)

    def calculate_f1_score(self):
        return round(metrics.f1_score(self.actual, self.predicted, average='weighted'),2)

    def get_performance(self):
        pre = round(1.0 * self.tp / (self.tp + self.fp),2) if (self.tp + self.fp) != 0 else 0
        rec = round(1.0 * self.tp / (self.tp + self.fn),2) if (self.tp + self.fn) != 0 else 0
        spec = round(1.0 * self.tn / (self.tn + self.fp),2) if (self.tn + self.fp) != 0 else 0
        fpr = 1 - spec
        npv = round(1.0 * self.tn / (self.tn + self.fn),2) if (self.tn + self.fn) != 0 else 0
        acc = round(1.0 * (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn),2) if (self.tp + self.tn + self.fp + self.fn) != 0 else 0
        f1 = round(2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn),2) if (2.0 * self.tp + self.fp + self.fn) != 0 else 0
        pd = round(1.0 * self.tp / (self.tp + self.fn),2)
        pf =  round(1.0 * self.fp / (self.fp + self.tn),2)
        return pre, rec, spec, fpr, npv, acc, f1,pd,pf

    def calculate_d2h(self):
        if (self.fp + self.tn) != 0:
            far = self.fp/(self.fp+self.tn)
        if (self.tp + self.fn) != 0:
            recall = self.tp/(self.tp + self.fn)
        dist2heaven = math.sqrt((1 - recall) ** 2 + far ** 2)
        print("dist",dist2heaven)
        return dist2heaven

    def get_g_score(self, beta = 0.5):
        g = (1 + beta**2) * (self.pd * (1.0 - self.pf))/ (beta ** 2 * self.pd + (1.0 - self.pf))
        return round(g,2)


    