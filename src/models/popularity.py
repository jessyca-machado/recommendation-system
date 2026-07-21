# import pandas as pd

# from .base import RecommenderBase


# class PopularityRecommender(RecommenderBase):

#     def __init__(self) -> None:
#         self._item_scores: pd.Series | None = None

#     def fit(self, interactions):

#         self.popular_items = (
#             interactions["item_idx"]
#             .value_counts()
#             .index
#             .tolist()
#         )

#         return self

#     def recommend(self, users, k=20):

#         predictions = []

#         for user in users:

#             for rank, item in enumerate(
#                 self.popular_items[:k]
#             ):

#                 predictions.append(
#                     [user, item, rank + 1]
#                 )

#         return pd.DataFrame(
#             predictions,
#             columns=[
#                 "user_idx",
#                 "item_idx",
#                 "rank"
#             ]
#         )

import pandas as pd
from .base import RecommenderBase

class PopularityRecommender(RecommenderBase):
    def __init__(self) -> None:
        self.item_scores: pd.Series | None = None

    def fit(self, interactions: pd.DataFrame):
        # use item_idx (seu pipeline 1)
        counts = interactions["item_idx"].value_counts()
        # score proporcional (normalização opcional)
        self.item_scores = counts / counts.max()
        return self

    def predict(self, user_ids: pd.Series, item_ids: pd.Series) -> pd.Series:
        if self.item_scores is None:
            raise RuntimeError("fit() antes de predict()")
        return item_ids.map(self.item_scores).fillna(0.0)

    def recommend(self, candidates: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """
        candidates: colunas [user_idx, item_idx] (um positivo + N negativos por user)
        retorna: [user_idx, item_idx, rank, score]
        """
        scores = self.predict(candidates["user_idx"], candidates["item_idx"])
        scored = candidates.copy()
        scored["score"] = scores

        scored = scored.sort_values(["user_idx", "score"], ascending=[True, False])
        scored["rank"] = scored.groupby("user_idx").cumcount() + 1
        return scored[scored["rank"] <= k][["user_idx", "item_idx", "rank", "score"]]
