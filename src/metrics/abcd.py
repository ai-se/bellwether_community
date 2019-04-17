from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from pdb import set_trace
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from typing import List, Tuple


class ABCD:
    def __init__(self,
                 actual: List[int],
                 predicted: List[int],
                 loc: List[int]) -> None:
        self.loc = pd.DataFrame(loc, columns=['file_loc'])
        self.actual = pd.DataFrame(actual, columns=['Actual'])
        self.predicted = pd.DataFrame(predicted, columns=['Predicted'])
        self.dframe = pd.concat(
            [self.actual, self.predicted, self.loc], axis=1)
        self.dframe = self.dframe.dropna()
        self.dframe = self.dframe.astype({'Actual': int, 'Predicted': int})
        self.dframe['InspectedLOC'] = self.dframe.file_loc.cumsum()
        self._set_aux_vars()

    def _set_aux_vars(self) -> None:
        """
        Set all the auxillary variables used for defect prediction
        """
        self.M = len(self.dframe)
        self.N = self.dframe.Actual.sum()
        inspected_max = self.dframe.InspectedLOC.max()
        for i in range(self.M):
            if self.dframe.InspectedLOC.iloc[i] >= 1 * inspected_max:
                # If we have inspected more than 20% of the total LOC
                # break
                break

        self.inspected_50 = self.dframe.iloc[:i]
        # Number of changes when we inspect 20% of LOC
        self.m = len(self.inspected_50)
        self.n = self.inspected_50.Predicted.sum()

    def get_pd_pf(self) -> Tuple[int, int]:
        """
        Obtain Recall (Pd) and False Alarm (Pf) scores

        Returns
        -------
        pd: float
            Recall (pd) value 
        pf: float
            False alarm (pf) values 
        """
        tn, fp, fn, tp = confusion_matrix(
            self.inspected_50.Actual, self.inspected_50.Predicted, labels=[0, 1]).ravel()

        pd = int(100 * tp / (tp + fn + 1e-5))
        pf = int(100 * fp / (fp + tn + 1e-5))

        return pd, pf

    def get_g_score(self, beta: int = 0.5) -> int:
        """
        Obtain G score

        Parameters
        ----------
        beta: float, default=1
            Amount by which recall (pd) is weighted higher than false alarm (pf)

        Returns
        -------
        g_: float
            G-Score
        """
        pd, pf = self.get_pd_pf()
        try:
            g_ = int((1 + beta**2) * (pd * (100 - pf))
                     / (beta ** 2 * pd + (100 - pf)))
            return g_
        except ZeroDivisionError:
            return 0

    def get_f_score(self, beta: int = 0.5) -> Tuple[int, int]:
        """
        Obtain F scores

        Parameters
        ----------
        beta: float, default=1
            Amount by which recall is weighted higher than precision

        Returns
        -------
        prec: float
            Precision
        f: float
            F score 
        """
        tn, fp, fn, tp = confusion_matrix(
            self.inspected_50.Actual, self.inspected_50.Predicted, labels=[0, 1]).ravel()
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        try:
            f = int(100 * (1 + beta**2) * (prec * recall)
                    / (beta ** 2 * prec + recall))
        except:
            return prec, 0

        prec = int(100 * prec)
        recall = int(100 * recall)
        return prec, f

    def get_pci_20(self) -> int:
        """
        Proportion of Changes Inspected when 20% LOC modified by all changes are 
        inspected. A high PCI@k% indicates that, under the same number of LOC to 
        inspect, developers need to inspect more changes.

        Returns
        -------
        int:
            The PCI value
        """
        pci_20 = int(self.m / self.M * 100)
        return pci_20

    def get_ifa(self) -> int:
        """
        Inital False Alarm

        Number of Initial False Alarms encountered before we find the first 
        defect. 

        Returns
        -------
        int:
            The IFA value

        Notes
        -----
        We compute the IFA by sorting the actual bug counts, and then computing
        the number of false alarms until the first true positive is discovered.

        The value is normalized to a percentage value.
        """

        for i in range(len(self.dframe)):
            if self.dframe['Actual'].iloc[i] == self.dframe['Predicted'].iloc[i] == 1:
                break

        pred_vals = self.dframe['Predicted'].values[:i]
        ifa = int(sum(pred_vals) / (i + 1) * 100)
        return i
