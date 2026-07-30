import joblib
import mlflow
from mlflow import MlflowClient

from src.config.settings import settings


def load_production_model(
    model_name: str,
    alias: str = "staging",
):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        name=model_name,
        alias=alias,
    )

    model_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_version.source,
    )

    return joblib.load(model_path)
