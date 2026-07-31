from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class RecommenderBase(ABC):
    """Interface base para modelos de recomendação."""

    @abstractmethod
    def fit(self, interactions: Any) -> "RecommenderBase":
        """Treina o modelo com os dados de interação.

        Args:
            interactions: Dados de interação usados no treinamento.

        Returns:
            RecommenderBase: Instância treinada do modelo.
        """

    @abstractmethod
    def recommend(self, candidates: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """Gera recomendações para os candidatos informados.

        Args:
            candidates: DataFrame com colunas user_idx e item_idx.
            k: Número máximo de itens recomendados por usuário.

        Returns:
            pd.DataFrame: Ranking com colunas user_idx, item_idx, rank e score.
        """
