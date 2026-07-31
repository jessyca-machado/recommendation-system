from typing import Any

import numpy as np
import pandas as pd
from implicit.bpr import BayesianPersonalizedRanking

from .base import RecommenderBase


class BPRRecommender(RecommenderBase):
    """Modelo BPR para recomendação baseada em pares."""

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.01,
        iterations: int = 50,
        random_state: int = 42,
    ) -> None:
        self.model = BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self.matrix: Any | None = None

    def fit(self, matrix: Any) -> "BPRRecommender":
        """Treina o modelo BPR com uma matriz esparsa.

        Args:
            matrix: Matriz esparsa de interações usuário-item.

        Returns:
            BPRRecommender: Instância treinada do modelo.
        """
        self.matrix = matrix.tocsr()
        self.model.fit(self.matrix)
        return self

    def recommend(self, candidates: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """Gera o top-k de itens por usuário.

        Args:
            candidates: DataFrame com colunas user_idx e item_idx.
            k: Quantidade máxima de recomendações por usuário.

        Returns:
            pd.DataFrame: Ranking com colunas user_idx, item_idx, rank e score.
        """
        user_factors = self.model.user_factors
        item_factors = self.model.item_factors

        preds: list[list[float | int]] = []
        for user, group in candidates.groupby("user_idx"):
            u = int(user)
            items = group["item_idx"].to_numpy(dtype=np.int32)

            scores = item_factors[items] @ user_factors[u]

            order = np.argsort(-scores)[:k]
            top_items = items[order]
            top_scores = scores[order]

            for rank, (item, score) in enumerate(
                zip(top_items, top_scores, strict=True),
                start=1,
            ):
                preds.append([u, int(item), rank, float(score)])

        return pd.DataFrame(
            preds,
            columns=["user_idx", "item_idx", "rank", "score"],
        )
