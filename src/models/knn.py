# src/models/knn.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .base import RecommenderBase


class KNNRecommender(RecommenderBase):
    def __init__(self, k_neighbors=50, precompute_neighbors=True, max_user_history=100):
        self.k_neighbors = k_neighbors
        self.precompute_neighbors = precompute_neighbors
        self.max_user_history = max_user_history

    def fit(self, matrix):
        self.matrix = matrix.tocsr()
        self.item_user_matrix = self.matrix.T.tocsr()

        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=self.k_neighbors,
            n_jobs=-1,
        ).fit(self.item_user_matrix)

        if self.precompute_neighbors:
            distances, indices = self.model.kneighbors(
                self.item_user_matrix, n_neighbors=self.k_neighbors
            )
            self.neigh_idx = indices.astype(np.int32)
            self.neigh_sim = (1.0 - distances).astype(np.float32)
        else:
            self.neigh_idx = None
            self.neigh_sim = None

        return self

    def recommend(self, candidates: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        preds = []

        # pré-agrupar candidatos por user (evita groupby overhead repetido)
        for user, group in candidates.groupby("user_idx", sort=False):
            user = int(user)
            consumed = self.matrix[user].indices
            if consumed.size == 0:
                continue

            # (muito importante) limita histórico gigante
            if self.max_user_history is not None and consumed.size > self.max_user_history:
                consumed = consumed[-self.max_user_history :]  # pega os últimos (ou amostra)

            # selected = group["item_idx"].to_numpy(dtype=np.int32)
            # selected_set = set(map(int, selected))
            # cand_scores = {int(i): 0.0 for i in selected}

            selected = group["item_idx"].to_numpy(dtype=np.int32)
            consumed_set = set(map(int, consumed))

            # remove consumidos dos candidatos
            selected_set = set(map(int, selected)) - consumed_set
            if not selected_set:
                continue

            cand_scores = {i: 0.0 for i in selected_set}

            if self.neigh_idx is not None:
                # usa vizinhos pré-computados
                for item_idx in consumed:
                    neigh_items = self.neigh_idx[item_idx]
                    neigh_sims = self.neigh_sim[item_idx]
                    for neigh_item, sim in zip(neigh_items, neigh_sims, strict=True):
                        neigh_item = int(neigh_item)
                        if neigh_item in selected_set:
                            cand_scores[neigh_item] += float(sim)
            else:
                # fallback: calcula on-the-fly (lento)
                for item_idx in consumed:
                    item_vector = self.item_user_matrix[item_idx]
                    distances, indices = self.model.kneighbors(
                        item_vector, n_neighbors=self.k_neighbors
                    )
                    for neigh_item, dist in zip(indices[0], distances[0], strict=True):
                        neigh_item = int(neigh_item)
                        if neigh_item in selected_set:
                            cand_scores[neigh_item] += 1.0 - float(dist)

            top = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            for rank, (item, score) in enumerate(top, start=1):
                preds.append([user, int(item), rank, float(score)])

        return pd.DataFrame(preds, columns=["user_idx", "item_idx", "rank", "score"])
