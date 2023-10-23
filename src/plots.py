import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from adjustText import adjust_text
import matplotlib.patheffects as pe
import os
import itertools
import collections
from datetime import datetime
import pandas as pd
import numpy as np


class Plots:
    """
    Class to visualize topic modeling related information.
    """

    def __init__(self, pipeline_instance):
        """
        Initialize the Plots class.

        Args:
            pipeline_instance (TopicModelPipeline): An instance of the TopicModelPipeline class.
        """
        self.pipeline_instance = pipeline_instance

        self.reduced_embeddings_2D = self.pipeline_instance.reduce_dimensionality(
            n_components=2
        )

    def plot_raw_clusters(self):
        """
        Plot the clusters in a 2D space.
        """
        plt.figure(figsize=(16, 16))

        unique_clusters = sorted(list(set(self.pipeline_instance.clusters)))

        for cluster in unique_clusters:
            plt.scatter(
                self.reduced_embeddings_2D[
                    self.pipeline_instance.clusters == cluster, 0
                ],
                self.reduced_embeddings_2D[
                    self.pipeline_instance.clusters == cluster, 1
                ],
                label=f"Cluster {cluster}",
                cmap="viridis",
                s=10,
            )

        plt.title("Clusters")
        plt.legend(loc="upper right")

        fig_path = os.path.join(self.pipeline_instance.project_path, "raw_clusters.png")
        plt.savefig(fig_path, dpi=300)

        plt.show()

    def plot_top_clusters(self):
        """
        Visualizes clusters by additionaly reducing embeddings to 2D.
        """

        df = pd.DataFrame(
            {
                "x": self.reduced_embeddings_2D[:, 0],
                "y": self.reduced_embeddings_2D[:, 1],
                "Topic": [str(t) for t in self.pipeline_instance.topic_model.topics_],
            }
        )

        unique_categories = df["Topic"].unique()
        category_colors = sns.color_palette("hsv", len(unique_categories)).as_hex()
        category_color_mapping = dict(zip(unique_categories, category_colors))
        df["Color"] = df["Topic"].map(category_color_mapping).astype(str)

        df["Length"] = [len(doc) for doc in self.pipeline_instance.doc_list]
        df = df.loc[df.Topic != "-1"]
        df = df.loc[(df.y > -30) & (df.y < 30) & (df.x < 30) & (df.x > -30), :]
        df["Topic"] = df["Topic"].astype("category")

        mean_df = df.groupby("Topic")[["x", "y"]].mean().reset_index()
        mean_df.Topic = mean_df.Topic.astype(int)
        mean_df = mean_df.sort_values("Topic")

        mean_df["Topic"] = mean_df["Topic"].astype(str)
        mean_df["Color"] = mean_df["Topic"].map(category_color_mapping).astype(str)
        mean_df["Color"].fillna("#faf7f7", inplace=True)

        fig = plt.figure(figsize=(16, 16))
        plt.scatter(
            x=df["x"],
            y=df["y"],
            c=df["Color"],
            alpha=0.4,
            s=df["Length"],
            cmap="viridis",
        )

        texts, xs, ys = [], [], []
        for index, row in mean_df.iterrows():
            topic = row["Topic"]
            name = " - ".join(
                list(zip(*self.pipeline_instance.topic_model.get_topic(int(topic))))[0][
                    :3
                ]
            )

            if int(topic) <= 50:
                xs.append(row["x"])
                ys.append(row["y"])
                texts.append(
                    plt.text(
                        row["x"],
                        row["y"],
                        name,
                        size=10,
                        ha="center",
                        color=row["Color"],
                        path_effects=[pe.withStroke(linewidth=0.5, foreground="black")],
                    )
                )

        adjust_text(
            texts,
            x=xs,
            y=ys,
            time_lim=1,
            force_text=(0.01, 0.02),
            force_static=(0.01, 0.02),
            force_pull=(0.5, 0.5),
        )

        fig_path = os.path.join(self.pipeline_instance.project_path, "top_clusters.png")
        fig.savefig(fig_path, dpi=300)

        return fig

    def plot_wordclouds(self):
        """
        Print wordcloud for each topic.
        """

        path = os.path.join(self.pipeline_instance.project_path, "Wordclouds")

        if not os.path.exists(path):
            os.makedirs(path)

        for topic in range(
            -1, (self.pipeline_instance.topic_model.get_topic_info().shape[0] - 1)
        ):
            text = {
                word: value
                for word, value in self.pipeline_instance.topic_model.get_topic(topic)
            }
            wc = WordCloud(background_color="white", max_words=1000)
            wc.generate_from_frequencies(text)
            plt.title(f"Wordcloud for topic {topic}")
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(os.path.join(path, f"wordcloud_topic_{topic}.png"))
            plt.show()
