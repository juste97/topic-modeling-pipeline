import os
import itertools
import collections
from datetime import datetime
import warnings
from numba import NumbaDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer


from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.representation import KeyBERTInspired

import torch

from umap import UMAP
from hdbscan import HDBSCAN

import transformers
from huggingface_hub import notebook_login
import joblib

from tqdm import tqdm
import locale

from src.dimensionality import Dimensionality
from src.preprocessor import Preprocessor
from src.plots import Plots


class TopicModelPipeline:
    def __init__(
        self,
        project_name: str,
        output_path: str,
        documents_path: str = "",
        file_path: str = "",
        file_type: str = None,
        model: str = "all-MiniLM-L6-v2",
        embeddings_path: str = "",
        reduced_embeddings_path: str = "",
        reduced_2D_embeddings_path: str = "",
        time_column: str = None,
        time_format: str = "%Y-%m-%d %H:%M:%S",
        text_column: str = None,
        sample: bool = False,
        sample_frequency: str = None,
        sample_by_column: str = None,
        clean_text: bool = False,
        vocab_frequency=0,
        min_samples=30,
        gen_min_span_tree=True,
        prediction_data=False,
        min_cluster_size=300,
        verbose=True,
        n_components=5,
        n_neighbors=10,
        metric="cosine",
        tresh=0,
        tresh_percent=1,
        tresh_absolut=100,
        blackwords: list = [],
        top_n_words: int = 10,
    ):
        """
        Args:
            project_name (str): Name of the project.
            path (str): Directory path for saving files.
            documents_path (str, optional): Path to the documents.
            file_path (str, optional): Path to read already merged file.
            file_type (str, optional): File type for reading.
            model (str, optional): Model name for SentenceTransformer. Defaults to "all-MiniLM-L6-v2".
            embeddings_path (str, optional): Path to embeddings.
            reduced_embeddings_path (str, optional): Path to reduced embeddings.
            reduced_2D_embeddings_path (str, optional): Path to 2D reduced embeddings.
            time_column (str, optional): Name of the column containing timestamps.
            text_column (str, optional): Name of the column containing text data.
            sample (bool, optional): Whether to sample the data.
            sample_frequency (str, optional): Frequency for sampling (e.g., 'day', 'month').
            sample_by_column (str, optional): Column name to sample by.
            clean_text (bool, optional): Whether to clean the text data.
            vocab_frequency (int, optional): Minimum frequency for a word to be included in the vocabulary.
            min_samples (int, optional): Minimum number of samples for HDBSCAN clustering.
            gen_min_span_tree (bool, optional): Whether to generate the minimum spanning tree for HDBSCAN.
            prediction_data (bool, optional): Whether to generate prediction data for HDBSCAN.
            min_cluster_size (int, optional): Minimum cluster size for HDBSCAN.
            verbose (bool, optional): Whether to display verbose output.
            n_components (int, optional): Number of components for UMAP dimensionality reduction.
            n_neighbors (int, optional): Number of neighbors for UMAP.
            metric (str, optional): Metric used for UMAP.
            tresh (int, optional): Threshold value for preprocessing to allow for heavily imbalanced data. Set to 0 to apply tresh_percent and tresh_absolut on every group.
            tresh_percent (float, optional): Threshold percentage for preprocessing.
            tresh_absolut (int, optional): Absolute threshold for preprocessing.
            blackwords (list, optional): List of words to be blacklisted during preprocessing.
            top_n_words (int, optional): Number of top words for each topic.
        """

        self.project_name = project_name
        self.output_path = output_path
        self.documents_path = documents_path
        self.file_path = file_path
        self.file_type = file_type
        self.embeddings_path = embeddings_path
        self.reduced_embeddings_path = reduced_embeddings_path
        self.reduced_2D_embeddings_path = reduced_2D_embeddings_path

        self.time_column = time_column
        self.time_format = time_format
        self.text_column = text_column

        self.sample = sample
        self.sample_frequency = sample_frequency
        self.sample_by_column = sample_by_column
        self.clean_text = clean_text
        self.tresh = tresh
        self.tresh_percent = tresh_percent
        self.tresh_absolut = tresh_absolut

        self.model = model
        self.vocab_frequency = vocab_frequency
        self.min_samples = min_samples
        self.gen_min_span_tree = gen_min_span_tree
        self.prediction_data = prediction_data
        self.min_cluster_size = min_cluster_size
        self.verbose = verbose
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.blackwords = blackwords
        self.top_n_words = top_n_words

        # checks
        if self.file_type is None and os.path.exists(self.documents_path):
            raise AssertionError(
                "You need to specify which type of files you want to load. Use: file_type."
            )

        if not os.path.exists(self.file_path) and not os.path.exists(
            self.documents_path
        ):
            raise AssertionError(
                "You either need to specify a file to load or a folder and the file format. Use: file_path or documents_path and file_type"
            )

        if self.sample:
            vars_to_check = {
                "sample_frequency": self.sample_frequency,
                "time_column": self.time_column,
                "tresh": self.tresh,
                "tresh_percent": self.tresh_percent,
                "tresh_absolut": self.tresh_absolut,
            }

            none_vars = [name for name, value in vars_to_check.items() if value is None]
            assert (
                not none_vars
            ), f"The following variables are None: {', '.join(none_vars)}"

        # set constants
        self.random_state = 42

        # everything else to be run
        self.make_project_folder()
        self.write_settings_file()

        self.doc_list, self.id_list = Preprocessor(
            self.file_path,
            self.file_type,
            self.documents_path,
            self.project_path,
            self.project_name,
            self.time_column,
            self.time_format,
            self.text_column,
            self.sample,
            self.sample_by_column,
            self.sample_frequency,
            self.tresh,
            self.tresh_percent,
            self.tresh_absolut,
            self.clean_text,
            self.random_state,
            self.blackwords,
        ).preprocess()

        self.plots = Plots(self)

        nltk.download("stopwords")
        self.stop_words = stopwords.words("english")

        self.model_fit()

    def make_project_folder(self):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")

        self.folder_name = f"{self.project_name}_{current_datetime}"
        self.project_path = os.path.join(self.output_path, self.folder_name)

        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
            print(f"Folder '{self.folder_name}' created!")
        else:
            print(f"Folder '{self.folder_name}' already exists!")

    def write_settings_file(self):
        param_dict = self.__dict__

        param_str = ""
        for key, value in param_dict.items():
            param_str += f"{key}: {value}\n"

        settings_filename = os.path.join(self.project_path, "settings.json")

        with open(settings_filename, "w") as file:
            file.write(param_str)

    def save_numpy(self, path, content):
        """
        Save numpy array to a file.

        Args:
            filename (str): Name of the file.
            content (array-like): Content to be saved.
        """
        filepath = path + ".npy"
        with open(filepath, "wb") as f:
            np.save(f, content)

    def load_numpy(self, path):
        """
        Load numpy array from a file.
        """

        return np.load(path)

    def tokenizer(self):
        """
        Tokenize the documents and build a vocabulary.
        """

        print("Tokenizing words and filtering for less used ones...")

        vocab = collections.Counter()
        tokenizer = CountVectorizer().build_tokenizer()
        for doc in tqdm(self.doc_list):
            vocab.update(tokenizer(doc))
        self.vocab = [
            word
            for word, frequency in vocab.items()
            if frequency >= self.vocab_frequency
        ]
        len(vocab)

    def encode_documents(self):
        """
        Encode the documents using SentenceTransformer.

        Args:
            device (str, optional): Device to be used for SentenceTransformer. Defaults to "cuda".
        """

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        filename = f"{self.project_name}_embeddings"
        full_path = os.path.join(self.project_path, filename)

        if os.path.exists(self.embeddings_path):
            print(f"File {self.embeddings_path} already exists. Loading embeddings...")
            self.embeddings = self.load_numpy(self.embeddings_path)
        else:
            print(f"Generating sentence level embeddings using {self.model}...")

            model = SentenceTransformer(self.model, device=device)
            self.embeddings = model.encode(self.doc_list, show_progress_bar=True)

            self.save_numpy(full_path, self.embeddings)

    def reduce_dimensionality(self, n_components=5):
        """
        Reduce the dimensionality of the embeddings using UMAP.

        Args:
            n_components (int): Number of dimensions to reduce the embeddings to.

        Returns:
            numpy.ndarray: Reduced embeddings.
        """

        filepath = os.path.join(
            self.project_path, f"{self.project_name}_reduced_embedding_{n_components}D"
        )

        if n_components == 2:
            reduced_embeddings_path = self.reduced_2D_embeddings_path
        else:
            reduced_embeddings_path = self.reduced_embeddings_path

        if os.path.exists(reduced_embeddings_path):
            print(
                f"File {reduced_embeddings_path} already exists. Loading reduced embedding data..."
            )
            reduced_embeddings = self.load_numpy(reduced_embeddings_path)
        else:
            print(f"Reducing embedding dimensionality to {n_components}D...")
            umap_model = UMAP(
                n_components=n_components,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                metric=self.metric,
                verbose=True,
            )
            reduced_embeddings = umap_model.fit_transform(self.embeddings)
            self.save_numpy(filepath, reduced_embeddings)

        return reduced_embeddings

    def cluster_documents(self):
        """
        Cluster the reduced embeddings using HDBSCAN.
        """

        filename = f"{self.project_name}_clusters"
        full_path = os.path.join(self.project_path, filename)

        if os.path.exists(full_path):
            print(f"File {filename} already exists. Loading cluster data...")
            self.clusters = self.load_numpy(full_path)
        else:
            print("Performing clustering on embeddings...")

            hdbscan_model = HDBSCAN(
                min_samples=self.min_samples,
                gen_min_span_tree=self.gen_min_span_tree,
                prediction_data=self.prediction_data,
                min_cluster_size=self.min_cluster_size,
            )
            self.clusters = hdbscan_model.fit(self.reduced_embeddings).labels_

            self.save_numpy(full_path, self.clusters)

    def prepare_models(self):
        """
        Prepare the necessary models for topic modeling.
        """

        self.tokenizer()
        self.encode_documents()
        self.embedding_model = SentenceTransformer(self.model)
        self.reduced_embeddings = self.reduce_dimensionality(n_components=5)
        self.cluster_documents()

        self.umap_model = Dimensionality(self.reduced_embeddings)
        self.hdbscan_model = BaseCluster()
        self.vectorizer_model = CountVectorizer(
            vocabulary=self.vocab, stop_words=self.stop_words
        )
        self.representation_model = KeyBERTInspired()

    def model_fit(self):
        """
        Fit the BERTopic model on the data.
        """

        self.prepare_models()

        print("Fitting the BERTopic model.")

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model,
            top_n_words=self.top_n_words,
            verbose=True,
        ).fit(self.doc_list, embeddings=self.embeddings, y=self.clusters)

        filepath = os.path.join(
            self.project_path, f"{self.project_name}_trained_topic_model"
        )
        self.topic_model.save(
            path=filepath,
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=self.model,
        )

        topics = self.topic_model.get_topic_info()

        topics.to_parquet(
            os.path.join(self.project_path, f"{self.project_name}_topic_info.parquet")
        )

    def plot_wordclouds(self):
        """
        Plot word clouds for the topics identified by the model.
        """
        self.plots.plot_wordclouds()

    def plot_clusters(self):
        """
        Plot clusters.
        """
        self.plots.plot_clusters()
