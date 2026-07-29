from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient


def register_model(
    model,
    model_name: str,
    artifact_path: str = "model",
) -> int:
    """
    Salva o modelo como artefato do run atual e cria uma nova versão
    no MLflow Model Registry.

    Returns:
        int: Número da versão criada.
    """

    model_file = Path("/tmp/model.joblib")
    joblib.dump(model, model_file)

    mlflow.log_artifact(
        str(model_file),
        artifact_path=artifact_path,
    )

    run_id = mlflow.active_run().info.run_id

    model_uri = f"runs:/{run_id}/{artifact_path}/model.joblib"

    client = MlflowClient()

    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    registered_model = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
    )

    return int(registered_model.version)


def promote_model(
    model_name: str,
    version: int,
    alias: str = "production",
):
    """
    Atribui um alias a uma versão registrada.
    """

    client = MlflowClient()

    client.set_registered_model_alias(
        name=str(model_name),
        alias=str(alias),
        version=int(version),
    )
