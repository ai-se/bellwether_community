"""
A data handler to read, write, and process data
"""
import os
import sys
import pandas as pd
from glob2 import glob
from pathlib import Path
from pdb import set_trace
from collections import OrderedDict

# Set path to src
root = Path.cwd()
while root.name is not 'src':
    # Climb up the directory tree until you reach
    root = root.parent

if root not in sys.path:
    sys.path.append(root)


class DataHandler:
    def __init__(self, data_path=root.joinpath("data")):
        """
        A Generic data handler class

        Parameters
        ----------
        data_path = <pathlib.PosixPath>
            Path to the data folder.
        """
        self.data_path = data_path

    def get_data(self, top_k=0):
        """
        Read data as pandas and return a dictionary of data

        Returns
        -------
        all_data: dict
            A dictionary of data with key-project_name, value-list of file
            level metrics
        """

        all_data = OrderedDict()
        projects = [file for file in self.data_path.glob("*.csv")]
        projects = sorted(
            projects, key=lambda file: file.stat().st_size, reverse=True)
        if top_k > 0:
            projects = projects[:top_k]

        for p in projects:
            temp_df = pd.read_csv(p)
            all_data.update(OrderedDict({p.stem: temp_df}))
        return all_data
