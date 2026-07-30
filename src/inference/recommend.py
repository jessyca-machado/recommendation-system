import pandas as pd

from src.config.settings import settings
from src.inference.loader import load_production_model

_model = None


def get_model():
    global _model

    if _model is None:
        _model = load_production_model(
            model_name=settings.mlflow_registered_model_name,
            alias="staging",
        )

    return _model


def recommend(
    candidates: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """
    Recebe um DataFrame de candidatos e delega
    a recomendação ao modelo carregado.
    """
    model = get_model()

    return model.recommend(
        candidates=candidates,
        k=k,
    )


def recommend_for_user(
    user_idx: int,
    items: list[int],
    k: int = 10,
) -> pd.DataFrame:
    """
    Monta os candidatos para um único usuário
    e retorna as Top-K recomendações.
    """
    candidates = pd.DataFrame(
        {
            "user_idx": [user_idx] * len(items),
            "item_idx": items,
        }
    )

    return recommend(
        candidates=candidates,
        k=k,
    )
