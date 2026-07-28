import numpy as np
import pandas as pd

from src.experiments.protocol import build_candidates_ncf


def evaluate_ncf_from_ranked(
    recs: pd.DataFrame,
    candidates: pd.DataFrame,
    k: int = 10,
) -> dict:
    """
    Avalia HR/NDCG/Precision/Recall no protocolo NCF (1 positivo por user).

    recs: DataFrame com colunas [user_idx, item_idx, rank] (top-k por user)
    candidates: DataFrame [user_idx, item_idx, label] com 1 positivo (label=1) por user
    """
    pos = candidates[candidates["label"] == 1][["user_idx", "item_idx"]]

    merged = pos.merge(
        recs[["user_idx", "item_idx", "rank"]],
        on=["user_idx", "item_idx"],
        how="left",
    )

    hit = merged["rank"].notna().astype(float)

    hit_rate = float(hit.mean())
    precision = float((hit / k).mean())

    r = merged["rank"].to_numpy(dtype=float)
    ndcg = float(np.where(np.isnan(r), 0.0, 1.0 / np.log2(r + 1)).mean())

    mrr = float(np.where(np.isnan(r), 0.0, 1.0 / r).mean())

    return {
        "hit_rate@k": hit_rate,
        "precision@k": precision,
        "ndcg@k": ndcg,
        "mrr@k": mrr,
    }


def evaluate_model_ncf(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k: int = 10,
    num_eval_negatives: int = 99,
    seed: int = 42,
    positive_strategy: str = "random_one",
) -> dict:
    candidates = build_candidates_ncf(
        test=test_df,
        train=train_df,
        n_items=n_items,
        num_eval_negatives=num_eval_negatives,
        seed=seed,
        positive_strategy=positive_strategy,
    )

    recs = model.recommend(
        candidates=candidates[["user_idx", "item_idx"]],
        k=k,
    )

    return evaluate_ncf_from_ranked(
        recs=recs,
        candidates=candidates,
        k=k,
    )
