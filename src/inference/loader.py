from typing import Any

import joblib
import mlflow
from mlflow import MlflowClient

from src.config.settings import settings


def load_production_model(
    model_name: str,
    alias: str = "staging",
) -> Any:
    """Carrega um modelo registrado no MLflow a partir de um alias.

    Args:
        model_name: Nome do modelo registrado no MLflow.
        alias: Alias associado à versão desejada.

    Returns:
        Any: Modelo carregado em memória.
    """

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
