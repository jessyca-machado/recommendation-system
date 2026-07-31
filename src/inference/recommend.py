from typing import Any

import pandas as pd


def recommend(
    model: Any,
    candidates: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Executa a recomendação de um modelo para um conjunto de candidatos.

    Args:
        model: Instância do modelo recomendador.
        candidates: DataFrame com colunas user_idx e item_idx.
        k: Número máximo de itens recomendados por usuário.

    Returns:
        pd.DataFrame: Ranking com colunas user_idx, item_idx, rank e score.
    """

    return model.recommend(
        candidates=candidates,
        k=k,
    )


def recommend_for_user(
    model: Any,
    item_catalog: pd.DataFrame,
    user_idx: int,
    k: int = 10,
) -> pd.DataFrame:
    """Gera recomendações para um usuário com base no catálogo de itens.

    Args:
        model: Instância do modelo recomendador.
        item_catalog: DataFrame com os itens disponíveis.
        user_idx: Identificador do usuário.
        k: Número máximo de recomendações.

    Returns:
        pd.DataFrame: Recomendação enriquecida com os dados do catálogo.
    """

    candidates = pd.DataFrame(
        {
            "user_idx": [user_idx] * len(item_catalog),
            "item_idx": item_catalog["item_idx"],
        }
    )

    recommendations = model.recommend(
        candidates=candidates,
        k=k,
    )

    recommendations = recommendations.merge(
        item_catalog,
        on="item_idx",
        how="left",
    )

    return recommendations
