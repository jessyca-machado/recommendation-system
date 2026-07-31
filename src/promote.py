# python src/promote.py

import json
import logging

import mlflow

from src.config.settings import settings
from src.experiments.registry import promote_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Promove o melhor modelo registrado para o alias de produção."""

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    with open(settings.models_dir / "best_model.json") as f:
        best_model = json.load(f)

    alias = "production"

    promote_model(
        model_name=settings.mlflow_registered_model_name,
        version=best_model["version"],
        alias=alias,
    )

    logger.info(
        'Model "%s" v%s promoted to "%s".',
        best_model["model"],
        best_model["version"],
        alias,
    )


if __name__ == "__main__":
    main()
