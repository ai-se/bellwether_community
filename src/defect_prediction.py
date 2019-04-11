"""
A Defect Prediction model for file level metrics
"""
import os
import re
import sys
import pandas
import numpy as np
from pathlib import Path
from pdb import set_trace
from prettytable import PrettyTable
from sklearn.model_selection import StratifiedKFold

root = Path.cwd()
# Climb up the directory tree until you reach
while root.name is not 'src':
    root = root.parent

if root not in sys.path:
    sys.path.append(root)

from metrics.abcd import ABCD
from datasets.data_handler import DataHandler
from prediction.model import PredictionModel

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    dh = DataHandler(data_path=root.joinpath("datasets"))
    mdl = PredictionModel()
    data = dh.get_data(top_k=25)

    # Create a Table than can pretty printed
    # --------------------------------------
    results = PrettyTable()
    results.field_names = ["Project", "    G", "   Pd", "   Pf", "   F1",
                           " Prec", "  IFA", "PCI20"]

    # Align Data
    # ----------
    results.align["Project"] = "l"
    results.align["    G"] = "r"
    results.align["   Pd"] = "r"
    results.align["   Pf"] = "r"
    results.align["   F1"] = "r"
    results.align["  IFA"] = "r"
    results.align[" Prec"] = "r"
    results.align["PCI20"] = "r"

    # Initialize K-Fold cross validation
    # ----------------------------------
    skfolds = StratifiedKFold(n_splits=5, random_state=0)

    for proj, dataset in data.items():
        print(proj.upper())

        # Separate out training and testing columns
        # -----------------------------------------
        indep = dataset[dataset.columns[:-1]]
        depen = dataset[dataset.columns[-1]]

        # Initialize datastructures to hold results of all folds
        # ------------------------------------------------------
        pd: list = []
        pf: list = []
        f1: list = []
        g_: list = []
        ifa: list = []
        prec: list = []
        pci20: list = []

        for _ in range(5):
            for train_index, test_index in skfolds.split(indep, depen):
                # Training data
                # -------------
                train_dataframe = dataset.iloc[train_index]

                # Testing data
                # ------------
                test_dataframe = dataset.iloc[test_index]

                # Build a prediction Model
                # ------------------------
                pred_mod = PredictionModel(classifier='SVC')

                # Make predictions
                # ----------------
                actual, predicted = pred_mod.predict_defects(
                    train_dataframe, test_dataframe)

                # Get lines of code of the test instances
                # ---------------------------------------
                loc = test_dataframe["file_loc"].tolist()
                abcd = ABCD(actual, predicted, loc)

                # Get performance metrics
                # -----------------------
                ifa_obtained = abcd.get_ifa()
                pci20_obtained = abcd.get_pci_20()
                pd_obtained, pf_obtained = abcd.get_pd_pf()
                g_score_obtained = abcd.get_g_score()
                prec_obtained, f1_obtained = abcd.get_f_score()

                # Save performance metrics
                # ------------------------
                pd.append(pd_obtained)
                pf.append(pf_obtained)
                f1.append(f1_obtained)
                ifa.append(ifa_obtained)
                prec.append(prec_obtained)
                g_.append(g_score_obtained)
                pci20.append(pci20_obtained)

        results.add_row([proj,
                         np.median(g_),
                         np.median(pd),
                         np.median(pf),
                         np.median(f1),
                         np.median(prec),
                         np.median(ifa),
                         np.median(pci20)])
    print(results)
