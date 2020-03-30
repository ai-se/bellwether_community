import copy
import math

from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class measures(object):

    def __init__(self,actual,predicted,loc,labels = [0,1]):
        self.actual = actual
        self.predicted = predicted
        self.loc = loc.values
        #self.dframe = pd.concat(
        #    [pd.Series(self.actual,name='Actual'), pd.Series(self.predicted,name='Predicted'), self.loc], axis=1)
        self.dframe = pd.DataFrame(list(zip(self.actual,self.predicted,self.loc)),columns = ['Actual','Predicted','LOC'])
        self.dframe = self.dframe.dropna()
        self.dframe = self.dframe.astype({'Actual': int, 'Predicted': int})
        self.dframe_unchanged = copy.deepcopy(self.dframe)
        self.dframe.sort_values(by = ['Predicted','LOC'],inplace=True,ascending=[False,True])
        self.dframe['InspectedLOC'] = self.dframe.LOC.cumsum()
        self.dframe_unchanged['InspectedLOC'] = self.dframe_unchanged.LOC.cumsum()
        self.tn, self.fp, self.fn, self.tp = metrics.confusion_matrix(
            actual, predicted, labels=labels).ravel()
        self.pre, self.rec, self.spec, self.fpr, self.npv, self.acc, self.f1,self.pd,self.pf = self.get_performance()
        #print(metrics.classification_report(self.actual,self.predicted))
        self._set_aux_vars()


    def _set_aux_vars(self):
        """
        Set all the auxillary variables used for defect prediction
        """
        self.M = len(self.dframe[self.dframe['Predicted'] == 1])
        self.N = self.dframe.Actual.sum() # have to check the implementation
        #inspected_max = self.dframe.InspectedLOC.max()
        inspected_max = self.dframe.InspectedLOC.max() * 0.2
        for i in range(self.M):
            if self.dframe.InspectedLOC.iloc[i] >= 1 * inspected_max:
                # If we have inspected more than 20% of the total LOC
                # break
                break
        if self.M == 0:
            i = 0
            self.M = 1
        self.inspected_50 = self.dframe.iloc[:i]
        # Number of changes when we inspect 20% of LOC
        self.m = len(self.inspected_50)
        self.n = self.inspected_50.Predicted.sum()

    
    def get_pci_20(self):
        pci_20 = self.m / self.M
        return round(pci_20,2)

    
    # def get_ifa(self):
    #     for i in range(len(self.dframe)):
    #         if self.dframe['Actual'].iloc[i] == self.dframe['Predicted'].iloc[i] == 1:
    #             break
    #     pred_vals = self.dframe['Predicted'].values[:i]
    #     ifa = int(sum(pred_vals) / (i + 1) * 100)
    #     return i

    def get_ifa(self):
        for i in range(len(self.dframe)):
            if self.dframe['Actual'].iloc[i] == self.dframe['Predicted'].iloc[i] == 1:
                break
        pred_vals = self.dframe['Predicted'].values[:i]
        ifa = int(sum(pred_vals) / (i + 1) * 100)
        return i
    
    def get_ifa_roc(self):
        ifa_x = []
        ifa_y = []
        for perc in range(1,101,1):
            count = 0
            inspected_max = self.dframe_unchanged.InspectedLOC.max() * (perc/100)
            for i in range(len(self.dframe_unchanged)):
                if self.dframe_unchanged.InspectedLOC.iloc[i] >= 1 * inspected_max:
                    break
                if self.dframe_unchanged['Predicted'].iloc[i] == 0:
                    continue
                count += 1 
                #if perc == 100:
                #    print(count,self.dframe_unchanged[self.dframe_unchanged['Predicted'] == 1].shape[0])
                if self.dframe_unchanged['Actual'].iloc[i] == self.dframe_unchanged['Predicted'].iloc[i] == 1:
                    break
            ifa_x.append(perc)
            ifa_y.append(count/self.dframe_unchanged[self.dframe_unchanged['Predicted'] == 1].shape[0])
        return np.trapz(ifa_y,x=ifa_x)
    
    
    def calculate_recall(self):
        if len(metrics.recall_score(self.actual, self.predicted, average=None)) == 1:
            if self.actual.unique()[0] == True:
                result = round(metrics.recall_score(self.actual, self.predicted, average=None)[0],2)
            else:
                result = 0
        else:
            result = round(metrics.recall_score(self.actual, self.predicted, average=None)[1],2)
        return result

    def calculate_precision(self):
        if len(metrics.precision_score(self.actual, self.predicted, average=None)) == 1:
            if self.actual.unique()[0] == True:
                result = round(metrics.precision_score(self.actual, self.predicted, average=None)[0],2)
            else:
                result = 0
        else:
            result = round(metrics.precision_score(self.actual, self.predicted, average=None)[1],2)
        return result

    def calculate_f1_score(self):
        if len(metrics.f1_score(self.actual, self.predicted, average=None)) == 1:
            if self.actual.unique()[0] == True:
                result = round(metrics.f1_score(self.actual, self.predicted, average=None)[0],2)
            else:
                result = 0
        else:
            result = round(metrics.f1_score(self.actual, self.predicted, average=None)[1],2)
        return result

    def get_performance(self):
        pre = round(1.0 * self.tp / (self.tp + self.fp),2) if (self.tp + self.fp) != 0 else 0
        rec = round(1.0 * self.tp / (self.tp + self.fn),2) if (self.tp + self.fn) != 0 else 0
        spec = round(1.0 * self.tn / (self.tn + self.fp),2) if (self.tn + self.fp) != 0 else 0
        fpr = round(1 - spec,2)
        npv = round(1.0 * self.tn / (self.tn + self.fn),2) if (self.tn + self.fn) != 0 else 0
        acc = round(1.0 * (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn),2) if (self.tp + self.tn + self.fp + self.fn) != 0 else 0
        f1 = round(2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn),2) if (2.0 * self.tp + self.fp + self.fn) != 0 else 0
        pd = round(1.0 * self.tp / (self.tp + self.fn),2)
        pf =  round(1.0 * self.fp / (self.fp + self.tn),2)
        return pre, rec, spec, fpr, npv, acc, f1,pd,pf

    def get_pd(self):
        return self.pd

    def get_pf(self):
        return self.pf

    def calculate_d2h(self):
        far = 0
        recall = 0
        if (self.fp + self.tn) != 0:
            far = self.fp/(self.fp+self.tn)
        if (self.tp + self.fn) != 0:
            recall = self.tp/(self.tp + self.fn)
        dist2heaven = math.sqrt((1 - recall) ** 2 + far ** 2)
        return round(dist2heaven,2)

    def get_g_score(self, beta = 0.5):
        g = (1 + beta**2) * (self.pd * (1.0 - self.pf))/ (beta ** 2 * self.pd + (1.0 - self.pf))
        return round(g,2)


    