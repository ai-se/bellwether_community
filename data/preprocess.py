import os
import re
import sys
import pandas as pd
from glob import glob
from pathlib import Path
from ipdb import set_trace
from typing import NoReturn
from collections import defaultdict

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


def get_buggy_clean_pairs(data_pkl: str) -> pd.DataFrame:
    """
    Find pairs of commits before and after bugfix

    Parameters
    ----------
    data_pkl: str
        Path to the data file

    Returns
    -------
    pd.core.frame.DataFrame:
        A data frame of only the buggy commits and it's ancestor
    """
    # Get the metadata file (this ends with _committed_file.pkl)
    # We can infer this dynamically from the data_pkl
    # -- Get parent --
    parent = data_pkl.parent
    # -- Get filname --
    old_f_name = data_pkl.name
    # -- Do some regex magic to change the filename --
    new_f_name = re.sub("_commit", "_committed_file", old_f_name)
    metadat_pkl = parent.joinpath(new_f_name)
    # Read the pickle files
    commits_df = pd.read_pickle(data_pkl)
    metadat_df = pd.read_pickle(metadat_pkl)
    metadat_df = metadat_df.drop(labels=['file_id', 'file_mode'], axis=1)
    metadat_df = metadat_df.set_index('commit_id').stack()
    all_pairs = []
    for i in range(len(commits_df)):
        if commits_df.iloc[i].buggy == 1:
            buggy = commits_df.iloc[i].parent
            clean = commits_df.iloc[i].commit_number
            if metadat_df.index.contains(clean):
                files_changed = metadat_df.loc[clean].values.tolist()
                all_pairs.append({
                    "clean": clean,
                    "buggy": buggy,
                    "changed": files_changed
                })
    return all_pairs


def preprocess_data_main() -> NoReturn:
    """
    Process Raw Data into csv
    """
    commits_path = Path("commits_labelled")
    pkl_files = commits_path.glob("*_commit.pkl")
    clean_and_buggy = defaultdict(list)
    for data_pkl in pkl_files:
        # -- Get project names --
        proj_name = data_pkl.with_suffix('').name[-7]
        # -- Get clean/buggy commit pairs --
        proj_clean_buggy_pairs = get_buggy_clean_pairs(data_pkl)
        # -- Save Results --
        clean_and_buggy[proj_name] = proj_clean_buggy_pairs
        set_trace()


if __name__ == "__main__":
    preprocess_data_main()
