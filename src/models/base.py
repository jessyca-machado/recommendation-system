from abc import ABC, abstractmethod
import pandas as pd

class RecommenderBase(ABC):

    @abstractmethod
    def fit(self, interactions):
        pass

    @abstractmethod
    def recommend(self, candidates: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        pass
