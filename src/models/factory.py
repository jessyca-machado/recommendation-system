# src/models/factory.py

from src.models.als import ALSRecommender
from src.models.bpr import BPRRecommender
from src.models.knn import KNNRecommender
from src.models.mlp import MLPRecommender
from src.models.popularity import PopularityRecommender

MODEL_FACTORY = {
    "popularity": {
        "constructor": PopularityRecommender,
        "input": "df",
    },
    "als": {
        "constructor": ALSRecommender,
        "input": "matrix",
    },
    "bpr": {
        "constructor": BPRRecommender,
        "input": "matrix",
    },
    "knn": {
        "constructor": KNNRecommender,
        "input": "matrix",
    },
    "mlp": {
        "constructor": MLPRecommender,
        "input": "matrix",
    },
}


def build_models(config: dict):
    models = {}

    for model_name, model_cfg in config.items():
        if not model_cfg["enabled"]:
            continue

        if model_name not in MODEL_FACTORY:
            raise ValueError(f"Unknown model: {model_name}")

        metadata = MODEL_FACTORY[model_name]

        kwargs = {k: v for k, v in model_cfg.items() if k != "enabled"}

        models[model_name] = {
            "model": metadata["constructor"](**kwargs),
            "input": metadata["input"],
        }

    return models
