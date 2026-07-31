import pandas as pd

from .base import RecommenderBase


class PopularityRecommender(RecommenderBase):
    """Modelo baseado em popularidade de itens."""

    def __init__(self) -> None:
        self.item_scores: pd.Series | None = None

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        """Treina o modelo de popularidade a partir das interações.

        Args:
            interactions: DataFrame com as interações usuário-item.

        Returns:
            PopularityRecommender: Instância treinada do modelo.
        """
        counts = interactions["item_idx"].value_counts()
        self.item_scores = counts / counts.max()
        return self

    def predict(self, user_ids: pd.Series, item_ids: pd.Series) -> pd.Series:
        """Calcula scores de popularidade para os pares usuário-item.

        Args:
            user_ids: Série com identificadores de usuários.
            item_ids: Série com identificadores de itens.

        Returns:
            pd.Series: Scores de popularidade para cada item.
        """
        if self.item_scores is None:
            raise RuntimeError("fit() antes de predict()")
        return item_ids.map(self.item_scores).fillna(0.0)

    def recommend(self, candidates: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """Gera recomendações com base na popularidade dos itens.

        Args:
            candidates: DataFrame com colunas user_idx e item_idx.
            k: Quantidade máxima de itens recomendados por usuário.

        Returns:
            pd.DataFrame: Ranking com colunas user_idx, item_idx, rank e score.
        """
        scores = self.predict(candidates["user_idx"], candidates["item_idx"])
        scored = candidates.copy()
        scored["score"] = scores

        scored = scored.sort_values(
            ["user_idx", "score"],
            ascending=[True, False],
        )
        scored["rank"] = scored.groupby("user_idx").cumcount() + 1
        return scored[scored["rank"] <= k][["user_idx", "item_idx", "rank", "score"]]
