from typing import Any

import numpy as np
import pandas as pd

from src.experiments.protocol import build_candidates_ncf


def evaluate_ncf_from_ranked(
    recs: pd.DataFrame,
    candidates: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Avalia métricas de ranking para o protocolo NCF.

    Args:
        recs: DataFrame com colunas user_idx, item_idx e rank.
        candidates: DataFrame com user_idx, item_idx e label.
        k: Tamanho do top-k avaliado.

    Returns:
        dict[str, float]: Métricas hit rate, precision, ndcg e mrr.
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
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k: int = 10,
    num_eval_negatives: int = 99,
    seed: int = 42,
    positive_strategy: str = "random_one",
) -> dict[str, float]:
    """Executa a avaliação de um modelo no protocolo NCF.

    Args:
        model: Modelo recomendador com método recommend.
        train_df: DataFrame de treino.
        test_df: DataFrame de teste.
        n_items: Quantidade total de itens.
        k: Tamanho do ranking avaliado.
        num_eval_negatives: Quantidade de negativos por usuário.
        seed: Semente aleatória para a geração de candidatos.
        positive_strategy: Estratégia para seleção dos positivos.

    Returns:
        dict[str, float]: Métricas de avaliação.
    """
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
