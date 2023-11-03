import mlflow
from datetime import datetime

# Start MLFLow server like this (replace path with mlruns folder in outputs folder):
# mlflow server --backend-store-uri file:C:\Users\steng\Github\topic-modeling-pipeline\experiments\output\mlflow


def start_run(cfg) -> None:
    """
    Start mlflow run.

    Args:
        cfg (DictConfig): Hydra config file.
    """
    tracking_uri = f"file:{cfg.mlflow.tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(cfg.experiment_name)

    run_name = cfg.experiment_name
    mlflow.start_run(
        run_name=(f'{run_name}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
    )


def end_run(cfg, metrics, raw_clusters, top_clusters) -> None:
    """
    End the mlflow run and track run information.

    Args:
        cfg (DictConfig): Hydra config file.
        metrics (Dictionary): Metrics to evaluate the clusters.
        raw_clusters (Plot): Raw cluster plot.
        top_clusters (Plot): Top cluster plot.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    pipeline_params = dict(cfg.pipeline)
    del pipeline_params["_target_"]
    for key, value in pipeline_params.items():
        mlflow.log_param(key, value)

    mlflow.log_dict(pipeline_params, "pipeline_parameters.json")
    # mlflow.log_figure(raw_clusters, "raw_clusters.png")
    # mlflow.log_figure(top_clusters, "top_clusters.png")

    mlflow.log_param("model_name", cfg.experiment_name)
    mlflow.end_run()
