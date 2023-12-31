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
        documents_path: str,
        project_path: str,
        project_name: str,
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
        random_state: int,
        blackwords: list = [],
    ):
        """
        Args:
            file_path (str): Path to read already merged file.
            file_type (str): File type for reading.
            documents_path (str): Path to the documents.
            project_path (str): Path to the project directory.
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
        self.documents_path = documents_path
        self.project_path = project_path
        self.project_name = project_name

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
        self.random_state = random_state
        self.blackwords = blackwords

        self.df_filename = f"{self.project_name}_merged_dataframe.parquet"

    def save_sampled_dataframe(self):
        """
        Save the sampled dataframe to a parquet file.
        """

        path = os.path.join(self.project_path, self.df_filename)
        self.df.to_parquet(path)

        print(f"Saving sampled dataframe with the name: {self.df_filename}.")

    def read_file(self, file_path):
        """
        Read a file based on its type and return the dataframe.

        Args:
            file_path (str): Path to the file to be read.

        Returns:
            pd.DataFrame: Pandas Dataframe.
        """
        file_type = file_path.split(".")[-1]

        if file_type == "parquet":
            return pd.read_parquet(file_path)
        elif file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "xlsx":
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def merge_parquet_files(self):
        """
        Merge multiple parquet files into a single dataframe.
        """
        files = glob.glob(f"{self.documents_path}/*.{self.file_type}")
        dfs = [self.read_file(f).assign(filename=f) for f in files]
        self.df = pd.concat(dfs, ignore_index=True)

    def convert_timestamps(self):
        """
        Convert timestamps in the dataframe to datetime format.
        """
        self.df[self.time_column] = self.df[self.time_column].swifter.apply(
            lambda x: pd.to_datetime(x, format=self.time_format)
        )

        # "%b %d, %Y · %I:%M %p UTC"

    def categorize_by_size(self):
        """
        Categorize the dataframe based on sample size.
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
        n_samples = min(int(len(group) * self.tresh_percent), self.tresh_absolut)
        return group.sample(n=n_samples, random_state=self.random_state)

    def sample_from_df(self):
        """
        Sample data from dataframe.

        """
        self.convert_timestamps()
        self.categorize_by_size()

        if self.sample_frequency == "day":
            time_grouping = self.many_samples[self.time_column].dt.day
        elif self.sample_frequency == "month":
            time_grouping = self.many_samples[self.time_column].dt.month
        else:
            time_grouping = self.many_samples[self.time_column].dt.isocalendar().week

        random_sampled = (
            self.many_samples.groupby([self.sample_by_column, time_grouping])
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
        pass

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
        trigger = False

        if os.path.exists(self.file_path):
            print(f"Loading dataframe {self.file_path}...")
            self.df = self.read_file(self.file_path)

        else:
            print("Loading, merging and cleaning files...")

            self.merge_parquet_files()

        if self.sample:
            print("Sampling the data....")

            if self.sample_by_column == None:
                self.sample_by_column = "Sample_Helper"
                self.df[self.sample_by_column] = 1

            self.sample_from_df()
            trigger = True

        if self.clean_text:
            print("Cleaning the text column...")
            self.clean_text_columns()
            trigger = True

        if trigger:
            self.save_sampled_dataframe()

        docs, ids = self.dataframe_to_list()

        return docs, ids
