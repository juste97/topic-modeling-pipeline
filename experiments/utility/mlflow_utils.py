import mlflow
from datetime import datetime

# Start MLFLow server like this (replace path with mlruns folder in outputs folder):
# mlflow server --backend-store-uri file:/Users/julian/Git/topic-modeling-pipeline/experiments/output/mlflow


def start_run(cfg) -> None:
    """
    Initializes an MLflow run using the given configuration.

    Parameters:
    - cfg (Config object): The configuration object containing experiment and model details.

    Returns:
    None
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
    Logs the results and model parameters to the current MLflow run and then ends the run.

    Parameters:
    - cfg (Config object): The configuration object containing model details.
    - results (dict): Dictionary containing the results/metrics to be logged.

    Returns:
    None
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    pipeline_params = dict(cfg.pipeline)
    del pipeline_params["_target_"]
    for key, value in pipeline_params.items():
        mlflow.log_param(key, value)

    mlflow.log_dict(pipeline_params, "pipeline_parameters.json")
    mlflow.log_figure(raw_clusters, "raw_clusters.png")
    mlflow.log_figure(top_clusters, "top_clusters.png")

    mlflow.log_param("model_name", cfg.experiment_name)
    mlflow.end_run()
