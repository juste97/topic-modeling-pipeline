import os
import re
import csv
import glob
import itertools
import collections
import time
from datetime import datetime
import pandas as pd
import swifter
import numpy as np


class Preprocessor:
    """
    Pipeline to clean and sample Twitter data from a dataframe.
    """

    def __init__(
        self,
        file_path: str,
        file_type: str,
        docs_path: str,
        project_path: str,
        project_name: str,
        path: str,
        time_column: str,
        time_format: str,
        text_column: str,
        sample: bool,
        sample_by_column: str,
        sample_frequency: str,
        tresh: int,
        tresh_percent: float,
        tresh_absolut: int,
        clean_text: bool,
        blackwords: list = [],
    ):
        """
        Args:
            file_path (str): Path to read already merged file.
            file_type (str): File type for reading.
            docs_path (str): Path to the documents.
            project_path (str): Path to the project directory.
            path (str): General directory path.
            time_column (str): Name of the column containing timestamps.
            text_column (str): Name of the column containing text data.
            sample (bool): Whether to sample the data.
            sample_by_column (str): Column name to sample by.
            sample_frequency (str): Frequency for sampling (e.g., 'day', 'month'. Default: "weekly").
            tresh (int): Threshold value for categorizing samples.
            tresh_percent (float): Weekly threshold percentage for sampling.
            tresh_absolut (int): Absolute weekly threshold for sampling.
            clean_text (bool): Whether to clean the text data.
            blackwords (list, optional): List of words to be blacklisted. Defaults to an empty list.
        """

        self.file_path = file_path
        self.file_type = file_type
        self.docs_path = docs_path
        self.project_path = project_path
        self.project_name = project_name
        self.path = path

        self.time_column = time_column
        self.time_format = time_format
        self.text_column = text_column

        self.sample = sample
        self.sample_by_column = sample_by_column
        self.sample_frequency = sample_frequency
        self.tresh = tresh
        self.tresh_percent = tresh_percent
        self.tresh_absolut = tresh_absolut

        self.clean_text = clean_text
        self.blackwords = blackwords

        self.df_filename = f"{self.project_name}_merged_dataframe.parquet"

    def save_sampled_dataframe(self):
        """
        Save the sampled dataframe to a parquet file.
        """

        path = os.path.join(self.project_path, self.df_filename)
        self.df.to_parquet(path)

        print(f"Saving sampled dataframe with the name: {self.df_filename}.")

    def merge_parquet_files(self):
        """
        Merge multiple parquet files into a single dataframe.
        """
        files = glob.glob(f"{self.docs_path}/*.{self.file_type}")

        if self.file_type == "parquet":
            dfs = [pd.read_parquet(f).assign(filename=f) for f in files]
        elif self.file_type == "csv":
            dfs = [pd.read_csv(f).assign(filename=f) for f in files]
        elif self.file_type == "xlsx":
            dfs = [pd.read_excel(f).assign(filename=f) for f in files]

        self.df = pd.concat(dfs, ignore_index=True)

    def convert_timestamps(self):
        """
        Convert timestamps in the dataframe to datetime format.
        """
        self.df[self.time_column] = self.df[self.time_column].swifter.apply(
            lambda x: pd.to_datetime(x, format=self.time_format)
        )

        #"%b %d, %Y · %I:%M %p UTC"

    def categorize_by_filename(self):
        """
        Categorize the dataframe based on the filename and sample size.
        """
        self.few_samples = self.df.groupby(self.sample_by_column).filter(
            lambda x: len(x) <= self.tresh
        )
        self.many_samples = self.df.groupby(self.sample_by_column).filter(
            lambda x: len(x) > self.tresh
        )

    def take_sample(self, group):
        """
        Sample data weekly based on the set thresholds.

        Args:
            group (DataFrame): Grouped dataframe.

        Returns:
            DataFrame: Sampled dataframe.
        """
        n_samples = min(
            int(len(group) * self.tresh_percent), self.tresh_absolut
        )
        return group.sample(n=n_samples)

    def sample_from_df(self):
        """
        Sample data from dataframe.

        """
        self.convert_timestamps()
        self.categorize_by_filename()

        if self.sample_frequency == 'day':
            time_grouping = self.many_samples[self.time_column].dt.day
        elif self.sample_frequency == 'month':
            time_grouping = self.many_samples[self.time_column].dt.month
        else:
            time_grouping = self.many_samples[self.time_column].dt.isocalendar().week

        random_sampled = (
            self.many_samples.groupby(
                [self.sample_by_column, time_grouping]
            )
            .apply(self.take_sample)
            .reset_index(drop=True)
        )
        self.sampled_df = pd.concat([self.few_samples, random_sampled])
        self.sampled_df.drop_duplicates(subset=self.text_column, inplace=True)

        self.df = self.sampled_df

    def clean_text_columns(self):
        """
        Clean the text columns in the dataframe.
        """
        if self.blackwords:
            replacement_dict = {
                r"(?i)" + word: "Institution" for word in self.blackwords
            }
            self.df[self.text_column] = self.df[self.text_column].replace(replacement_dict, regex=True)

        self.df[self.text_column] = self.df[self.text_column].swifter.apply(
            lambda x: x.strip().replace("\n", "").replace("'", "'")
        )
        self.df[self.text_column] = self.df[self.text_column].replace(r"@\w+", "", regex=True)

    def dataframe_to_list(self):
        """
        Convert the dataframe columns to lists.

        Returns:
            tuple: A tuple containing lists of documents and their corresponding IDs.
        """
        docs = self.df[self.text_column].tolist()
        ids = self.df[self.text_column].tolist()
        return docs, ids

    def preprocess(self):
        """
        Execute the preprocessing pipeline.

        Returns:
            tuple: A tuple containing lists of preprocessed documents and their corresponding IDs.
        """

        if not self.file_path is None:
            print(f"File {self.file_path} already exists. Loading dataframe...")
            self.df = pd.read_parquet(self.file_path)

        else:

            print("Loading, merging and cleaning files.")

            self.merge_parquet_files()

            if self.sample:
                print("Taking a subsample of the data...")

                if self.sample_by_column == None:

                    self.sample_by_column = "Sample_Helper"
                    self.df[self.sample_by_column] = 1

                self.sample_from_df()

            if self.clean_text:
                print("Cleaning the text column...")
                self.clean_text_columns()

            self.save_sampled_dataframe()

        docs, ids = self.dataframe_to_list()

        return docs, ids