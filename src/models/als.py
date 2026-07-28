# src/models/als.py

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

from src.models.base import RecommenderBase


class ALSRecommender(RecommenderBase):
    def __init__(
        self,
        factors=64,
        iterations=20,
        regularization=0.01,
    ):
        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
        )

    def fit(self, matrix):
        self.user_item_matrix = matrix.tocsr()

        self.item_user_matrix = self.user_item_matrix.T.tocsr()

        self.model.fit(self.item_user_matrix)

        return self

    def recommend(
        self,
        candidates: pd.DataFrame,  # colunas: user_idx, item_idx
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Rankeia somente itens candidatos por usuário (protocolo NCF).
        Retorna: user_idx, item_idx, rank, score
        """
        preds = []
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

        return pd.DataFrame(preds, columns=["user_idx", "item_idx", "rank", "score"])
