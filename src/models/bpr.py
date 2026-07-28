import numpy as np
import pandas as pd
from implicit.bpr import BayesianPersonalizedRanking

from .base import RecommenderBase


class BPRRecommender(RecommenderBase):
    def __init__(
        self, factors=64, learning_rate=0.01, regularization=0.01, iterations=50, random_state=42
    ):
        self.model = BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
        )
        self.matrix = None

    def fit(self, matrix):
        self.matrix = matrix.tocsr()
        self.model.fit(self.matrix)
        return self

    def recommend(self, candidates: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """
        candidates: colunas [user_idx, item_idx]
        retorna top-k por usuário dentro dos candidatos.
        """
        user_factors = self.model.user_factors
        item_factors = self.model.item_factors

        preds = []
        for user, group in candidates.groupby("user_idx"):
            u = int(user)
            items = group["item_idx"].to_numpy(dtype=np.int32)

            # score = dot(user_vec, item_vec)
            scores = item_factors[items] @ user_factors[u]

            order = np.argsort(-scores)[:k]
            top_items = items[order]
            top_scores = scores[order]

            for rank, (item, score) in enumerate(zip(top_items, top_scores, strict=True), start=1):
                preds.append([u, int(item), rank, float(score)])

        return pd.DataFrame(preds, columns=["user_idx", "item_idx", "rank", "score"])
