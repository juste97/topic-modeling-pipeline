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


    def plot_clusters(self):
        """
        Visualizes clusters by additionaly reducing embeddings to 2D.
        """

        self.pipeline_instance.reduce_dimensionality_2D()

        colors = itertools.cycle(
            [
                "#e6194b",
                "#3cb44b",
                "#ffe119",
                "#4363d8",
                "#f58231",
                "#911eb4",
                "#46f0f0",
                "#f032e6",
                "#bcf60c",
                "#fabebe",
                "#008080",
                "#e6beff",
                "#9a6324",
                "#fffac8",
                "#800000",
                "#aaffc3",
                "#808000",
                "#ffd8b1",
                "#000075",
                "#808080",
                "#ffffff",
                "#000000",
            ]
        )
        color_key = {
            str(topic): next(colors)
            for topic in set(self.pipeline_instance.topic_model.topics_)
            if topic != -1
        }

        df = pd.DataFrame(
            {
                "x": self.pipeline_instance.reduced_embeddings_2D[:, 0],
                "y": self.pipeline_instance.reduced_embeddings_2D[:, 1],
                "Topic": [str(t) for t in self.pipeline_instance.topic_model.topics_],
            }
        )
        df["Length"] = [len(doc) for doc in self.pipeline_instance.doc_list]
        df = df.loc[df.Topic != "-1"]
        df = df.loc[(df.y > -10) & (df.y < 10) & (df.x < 10) & (df.x > -10), :]
        df["Topic"] = df["Topic"].astype("category")

        mean_df = df.groupby("Topic").mean().reset_index()
        mean_df.Topic = mean_df.Topic.astype(int)
        mean_df = mean_df.sort_values("Topic")

        fig = plt.figure(figsize=(16, 16))
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            c=df["Topic"].map(color_key),
            alpha=0.4,
            sizes=(0.4, 10),
            size="Length",
        )

        texts, xs, ys = [], [], []
        for row in mean_df.iterrows():
            topic = row[1]["Topic"]
            name = " - ".join(list(zip(*self.pipeline_instance.topic_model.get_topic(int(topic))))[0][:3])

            if int(topic) <= 50:
                xs.append(row[1]["x"])
                ys.append(row[1]["y"])
                texts.append(
                    plt.text(
                        row[1]["x"],
                        row[1]["y"],
                        name,
                        size=10,
                        ha="center",
                        color=color_key[str(int(topic))],
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

        fig_path = os.path.join(self.pipeline_instance.project_path, "output_filename.png")
        fig.savefig(fig_path, dpi=300)

        return fig
    
    def plot_wordclouds(self):
        """
        Print wordcloud for each topic.
        """
        for topic in range(-1,(self.pipeline_instance.topic_model.get_topic_info().shape[0]-1)):
            text = {word: value for word, value in self.pipeline_instance.topic_model.get_topic(topic)}
            wc = WordCloud(background_color="white", max_words=1000)
            wc.generate_from_frequencies(text)
            topic = ""
            plt.title(f"Wordcloud for topic {topic}")
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.show()