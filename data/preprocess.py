import os
import re
import sys
import pandas as pd
from glob import glob
from pathlib import Path
from ipdb import set_trace
from typing import List
from collections import defaultdict

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


class Preprocess:
    """
    A Data preprocessor that takes the following data to create a defect dataset
    1. Labelled commits (in data/commits_labeled)
    2. Complexity metrics (in data/git_mine)
    3. Static code violations (in data/plato)
    """
    @staticmethod
    def _get_buggy_clean_pairs(data_pkl: Path) -> List[dict]:
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
        all_pairs: List[dict] = []
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

    @staticmethod
    def _dedupe(dframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that have the exact same values for everything except for 
        defects labels

        Parameters
        ----------
        dframe: pd.DataFrame

        Returns
        -------

        """
        # Relevant columns
        cols = [col for col in dframe.columns if not col in [
            'Unnamed: 0', 'BUGS']]
        # Drop duplicate rows
        # Remove both because we are not sure
        dframe = dframe.drop_duplicates(subset=cols, keep=False)
        return dframe

    def _build_dataset(self, proj_name: str, clean_buggy_commits: List[dict]) -> pd.DataFrame:
        """
        Gather the metrics and violations from the respective folders and popluate
        them in a pandas dataframe with appropriate lables

        Parameters
        ----------
        proj_name: str
            Name of the project being processed
        clean_buggy_commits: List[dict]
            Pairs of clean and buggy commits to be processed

        Returns
        -------
        pd.DataFrame:
            The entire dataset as a pandas dataframe
        """
        dframe: pd.DataFrame = pd.DataFrame()
        for item in clean_buggy_commits:
            # Only do something if changed files contains a .js file
            if any(map(lambda f: f.endswith(".js"), item['changed'])):
                # -- Complexity metrics --

                # --- Clean file ---
                clean_file_path_complex = Path("git_mine").joinpath(
                    proj_name).glob(item["clean"] + ".csv")

                clean_file_path_staticc = Path("plato").joinpath(
                    proj_name).glob(item["clean"] + ".csv")

                # --- Buggy file file ---
                buggy_file_path_complex = Path("git_mine").joinpath(
                    proj_name).glob(item["buggy"] + ".csv")

                buggy_file_path_staticc = Path("plato").joinpath(
                    proj_name).glob(item["buggy"] + ".csv")

                for clean_csv, clean_csv_staticc, buggy_csv, buggy_csv_staticc in zip(
                        clean_file_path_complex,
                        clean_file_path_staticc,
                        buggy_file_path_complex,
                        buggy_file_path_staticc):

                    # ---- Clean dataframe ----
                    clean_df = pd.read_csv(clean_csv)
                    clean_df_staticc = pd.read_csv(clean_csv_staticc)
                    # Fix filenames
                    clean_df_staticc['Unnamed: 0'] = clean_df_staticc['Unnamed: 0'].apply(
                        lambda x: x.split('source-code')[-1])
                    clean_df = pd.concat(
                        [clean_df, clean_df_staticc], axis=1, join="inner")
                    clean_df['BUGS'] = 0
                    dframe = dframe.append(clean_df, ignore_index=True)

                    # ---- Buggy dataframe ----
                    buggy_df = pd.read_csv(buggy_csv)
                    buggy_df_staticc = pd.read_csv(buggy_csv_staticc)
                    buggy_df = pd.concat(
                        [buggy_df, buggy_df_staticc], axis=1, join="inner")
                    buggy_df['BUGS'] = 1
                    dframe = dframe.append(buggy_df, ignore_index=True)

        dframe = self._dedupe(dframe)
        return dframe

    def main(self) -> None:
        """
        Process Raw Data into csv
        """
        commits_path = Path("commits_labeled")
        pkl_files = commits_path.glob("*_commit.pkl")
        for data_pkl in pkl_files:
            # -- Get project names --
            proj_name = data_pkl.stem[:-7]
            try:
                # -- Get clean/buggy commit pairs --
                proj_clean_buggy_pairs = self._get_buggy_clean_pairs(data_pkl)
                # -- Build the dataset --
                dataset = self._build_dataset(
                    proj_name, proj_clean_buggy_pairs)
            except:
                print(proj_name)
                continue
            if dataset.empty:
                print(proj_name)
                continue

            # -- Drop filename --
            dataset.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
            # -- Save Dataset --
            save_name = Path.cwd().parent.joinpath(
                'src', 'data', proj_name).with_suffix('.csv')
            dataset.to_csv(save_name, index=False)


if __name__ == "__main__":
    preproc = Preprocess()
    preproc.main()
