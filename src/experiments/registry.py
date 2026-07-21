# src/experiments/registry.py

from src.models.bpr import BPRRecommender
from src.models.popularity import PopularityRecommender
from src.models.knn import KNNRecommender
from src.models.als import ALSRecommender
from src.models.mlp import MLPRecommender

MODEL_REGISTRY = {
    "popularity": {
        "model": PopularityRecommender,
        "input": "df"
    },
    "knn": {
        "model": KNNRecommender,
        "input": "matrix"
    },
    "als": {
        "model": ALSRecommender,
        "input": "matrix"
    },
    "mlp": {
        "model": MLPRecommender,
        "input": "matrix"
    },
    "bpr": {
        "model": BPRRecommender,
        "input": "matrix"
    },
}
