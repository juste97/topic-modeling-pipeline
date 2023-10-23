import sklearn
from sklearn.metrics import *
import numpy as np


class Evaluator:
    def __init__(
        self,
        X: np.array,
        model: np.array,
        sil_score_metric: str = "euclidean",
        path: str = "",
    ):
        self.X = X
        self.model = model
        self.sil_score_metric = sil_score_metric
        self.path = path
        self.labels = self.model.labels_

        """
        Args:
            X (np.array): The input data samples.
            model (np.array): Fitted clustering model.
            sil_score_metric (str, optional): Metric to use when calculating the silhouette score. Defaults to "euclidean".
            path (str, optional): Path for saving raw cluster plot.
        """

    def reshape_data(self):
        """
        Reshapes the data if necessary.
        """
        pass

    def silhouette_score(self):
        """
        Calculate the silhouette score for the current clustering.
        The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample.
        The Silhouette Coefficient for a sample is (b - a) / max(a, b).
        https://www.sciencedirect.com/science/article/pii/0377042787901257

        Returns:
            float: Silhouette score.
        """
        return sklearn.metrics.silhouette_score(
            self.X, self.labels, metric=self.sil_score_metric
        )

    def davies_bouldin_score(self):
        """
        Calculate the Davies-Bouldin score for the current clustering.
        The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances.
        https://ieeexplore.ieee.org/document/4766909

        Returns:
            float: Davies-Bouldin score.
        """
        return sklearn.metrics.davies_bouldin_score(self.X, self.labels)

    def calinski_harabasz_score(self):
        """
        Calculate the Calinski-Harabasz score for the current clustering.
        The score is defined as ratio of the sum of between-cluster dispersion and of within-cluster dispersion.
        https://www.tandfonline.com/doi/abs/10.1080/03610927408827101

        Returns:
            float: Calinski-Harabasz score.
        """
        return sklearn.metrics.calinski_harabasz_score(self.X, self.labels)

    def calculate_metrics(self):
        """
        Calculate multiple metrics for evaluating the clustering.

        Returns:
            dict: Dictionary containing silhouette score, Davies-Bouldin score, and Calinski-Harabasz score.
        """
        metrics = {}
        metrics["silhouette_score"] = self.silhouette_score()
        metrics["davies_bouldin_score"] = self.davies_bouldin_score()
        metrics["calinski_harabasz_score"] = self.calinski_harabasz_score()

        return metrics
