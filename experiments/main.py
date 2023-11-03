import hydra
from omegaconf import DictConfig
import numpy as np

import sys

# why doesnt sys.append.path("..") work?
sys.path.append("/Users/julian/Git/topic-modeling-pipeline")

from src.topic_model import *
from experiments.utility.mlflow_utils import *


import numpy as np


@hydra.main(
    version_base="1.2",
    config_path=r"/Users/julian/Git/topic-modeling-pipeline/experiments/configs",
    config_name="config",
)
def main(cfg: DictConfig):
    start_run(cfg)

    model = hydra.utils.instantiate(cfg.pipeline)

    model.topic_model.get_topic_info()
    raw_clusters = model.plot_raw_clusters()
    top_clusters = model.plot_top_clusters()

    metrics = model.metrics
    optuna_metric = metrics[cfg.parameter]

    print(f"This trials result is: {optuna_metric}")

    end_run(cfg, metrics, raw_clusters, top_clusters)

    return optuna_metric


if __name__ == "__main__":
    main()
