import copy
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset

from .base import RecommenderBase

logger = logging.getLogger("runner")


class InteractionDataset(Dataset):
    def __init__(self, interactions: pd.DataFrame):
        self.users = interactions["user_idx"].values
        self.items = interactions["item_idx"].values
        self.labels = interactions["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class MLP(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, emb_dim)

        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)

        x = torch.cat([user_emb, item_emb], dim=1)

        return self.fc(x).squeeze(-1)


class MLPRecommender(RecommenderBase):
    def __init__(
        self,
        emb_dim: int = 16,
        lr: float = 1e-3,
        epochs: int = 25,
        batch_size: int = 2048,
        negative_ratio: int = 1,
        patience: int = 5,
        dropout: float = 0.2,
        seed: int = 42,
        device: str | None = None,
    ):
        self.emb_dim = emb_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.patience = patience
        self.dropout = dropout

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.rng = np.random.default_rng(seed)

        torch.manual_seed(seed)

    def fit(
        self,
        train_matrix: csr_matrix,
    ):
        self.train_matrix = train_matrix

        users, items = train_matrix.nonzero()

        train_interactions = pd.DataFrame(
            {
                "user_idx": users,
                "item_idx": items,
            }
        )

        self.user_items = train_interactions.groupby("user_idx")["item_idx"].apply(set).to_dict()

        self.n_users = train_matrix.shape[0]
        self.n_items = train_matrix.shape[1]

        self.model = MLP(
            n_users=self.n_users,
            n_items=self.n_items,
            emb_dim=self.emb_dim,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

        best_loss = float("inf")

        best_weights = copy.deepcopy(self.model.state_dict())

        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            train_df = self._build_training_dataframe(train_interactions)

            train_loader = DataLoader(
                InteractionDataset(train_df),
                batch_size=self.batch_size,
                shuffle=True,
            )

            train_loss = self._train_epoch(
                train_loader,
                criterion,
                optimizer,
            )

            logger.info(f"Epoch {epoch + 1:02d} | " f"loss={train_loss:.4f}")

            if train_loss < best_loss:
                best_loss = train_loss

                best_weights = copy.deepcopy(self.model.state_dict())

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping " f"after {epoch + 1} epochs")

                    break

        self.model.load_state_dict(best_weights)

        return self

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        self.model.train()

        total_loss = 0.0

        for user, item, label in loader:
            user = user.to(self.device)
            item = item.to(self.device)
            label = label.to(self.device)

            optimizer.zero_grad()

            preds = self.model(
                user,
                item,
            )

            loss = criterion(
                preds,
                label,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        self.model.eval()

        total_loss = 0.0

        with torch.no_grad():
            for user, item, label in loader:
                user = user.to(self.device)
                item = item.to(self.device)
                label = label.to(self.device)

                preds = self.model(
                    user,
                    item,
                )

                loss = criterion(
                    preds,
                    label,
                )

                total_loss += loss.item()

        return total_loss / len(loader)

    def _build_training_dataframe(
        self,
        interactions: pd.DataFrame,
    ) -> pd.DataFrame:
        positives = interactions[["user_idx", "item_idx"]].copy()

        positives["label"] = 1.0

        negatives = self._sample_negatives(positives)

        return pd.concat(
            [positives, negatives],
            ignore_index=True,
        )

    def _sample_negatives(
        self,
        positives: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []

        for user_id in positives["user_idx"]:
            consumed = self.user_items[user_id]

            sampled = set()

            while len(sampled) < self.negative_ratio:
                candidate = self.rng.integers(
                    0,
                    self.n_items,
                )

                if candidate in consumed:
                    continue

                if candidate in sampled:
                    continue

                sampled.add(candidate)

            for item_id in sampled:
                rows.append(
                    (
                        user_id,
                        item_id,
                        0.0,
                    )
                )

        return pd.DataFrame(
            rows,
            columns=[
                "user_idx",
                "item_idx",
                "label",
            ],
        )

    def predict(
        self,
        user_ids: pd.Series,
        item_ids: pd.Series,
    ) -> pd.Series:
        self.model.eval()

        users = torch.tensor(
            user_ids.values,
            dtype=torch.long,
        ).to(self.device)

        items = torch.tensor(
            item_ids.values,
            dtype=torch.long,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(
                users,
                items,
            )

            scores = torch.sigmoid(logits).cpu().numpy()

        return pd.Series(
            scores,
            index=user_ids.index,
        )

    def recommend(self, candidates: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        scored = candidates.copy()
        scored["score"] = self.predict(scored["user_idx"], scored["item_idx"])
        scored = scored.sort_values(["user_idx", "score"], ascending=[True, False])
        scored["rank"] = scored.groupby("user_idx").cumcount() + 1
        return scored[scored["rank"] <= k][["user_idx", "item_idx", "rank", "score"]]
