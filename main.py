import hydra
from omegaconf import DictConfig
import numpy as np
import sys
from src.topic_model import *
from experiments.utility.mlflow_utils import *


@hydra.main(
    version_base="1.2",
    config_path=r"C:\Users\steng\Github\topic-modeling-pipeline\experiments\configs",
    config_name="config",
)
def main(cfg: DictConfig) -> float:
    """
    Main function to start multirun for cluster experiments.

    Args:
        cfg (DictConfig): Hydra config file.

    Returns:
        optuna_metric (float): Metric (trial result) to be minimzed by optuna.
    """
    start_run(cfg)

    model = hydra.utils.instantiate(cfg.pipeline)

    model.topic_model.get_topic_info()
    top_clusters = model.plot_top_clusters()
    raw_clusters = model.plot_raw_clusters()

    metrics = model.metrics
    optuna_metric = metrics[cfg.parameter]

    print(f"This trials result is: {optuna_metric}")

    end_run(cfg, metrics, raw_clusters, top_clusters)

    return optuna_metric


if __name__ == "__main__":
    main()
