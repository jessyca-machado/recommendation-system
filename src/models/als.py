# src/models/als.py

from typing import Any

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

from src.models.base import RecommenderBase


class ALSRecommender(RecommenderBase):
    """Modelo ALS para recomendação de itens."""

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 20,
        regularization: float = 0.01,
    ) -> None:
        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
        )

    def fit(self, matrix: Any) -> "ALSRecommender":
        """Treina o modelo com uma matriz de interação esparsa.

        Args:
            matrix: Matriz esparsa de interações usuário-item.

        Returns:
            ALSRecommender: Instância treinada do modelo.
        """
        self.user_item_matrix = matrix.tocsr()

        self.item_user_matrix = self.user_item_matrix.T.tocsr()

        self.model.fit(self.item_user_matrix)

        return self

    def recommend(
        self,
        candidates: pd.DataFrame,
        k: int = 10,
    ) -> pd.DataFrame:
        """Gera ranking de itens para cada usuário a partir dos candidatos.

        Args:
            candidates: DataFrame com colunas user_idx e item_idx.
            k: Quantidade máxima de recomendação por usuário.

        Returns:
            pd.DataFrame: Ranking com colunas user_idx, item_idx, rank e score.
        """
        preds: list[list[float | int]] = []
        for user, group in candidates.groupby("user_idx"):
            user_items = self.user_item_matrix[user]
            selected = group["item_idx"].to_numpy(dtype=np.int32)

            item_ids, scores = self.model.rank_items(
                userid=int(user),
                user_items=user_items,
                selected_items=selected,
                recalculate_user=True,
            )

            top_item_ids = item_ids[:k]
            top_scores = scores[:k]

            for rank, (item, score) in enumerate(
                zip(top_item_ids, top_scores, strict=True), start=1
            ):
                preds.append([int(user), int(item), rank, float(score)])

        return pd.DataFrame(
            preds,
            columns=["user_idx", "item_idx", "rank", "score"],
        )
