# src/experiments/runner.py

import logging
import time

import pandas as pd
from rich.console import Console
from rich.table import Table
from scipy.sparse import csr_matrix

from src.experiments.evaluator import evaluate_model_ncf

logger = logging.getLogger("runner")
console = Console()


def run_experiments(
    train_df: pd.DataFrame,
    train_matrix: csr_matrix,
    test: pd.DataFrame,
    models: dict[str, dict[str, object]],
    k: int = 10,
    n_items: int | None = None,
    num_eval_negatives: int = 99,
    random_seed: int = 42,
) -> list[dict[str, object]]:
    """Executa treinamento e avaliação para todos os modelos configurados.

    Args:
        train_df: DataFrame com as interações de treino.
        train_matrix: Matriz esparsa de treino.
        test: DataFrame com as interações de teste.
        models: Configuração dos modelos a serem avaliados.
        k: Tamanho do top-k avaliado.
        n_items: Quantidade total de itens.
        num_eval_negatives: Quantidade de negativos para avaliação.
        random_seed: Semente aleatória para avaliação.

    Returns:
        list[dict[str, object]]: Lista com métricas e estimadores por modelo.
    """

    results: list[dict[str, object]] = []

    total_models = len(models)

    logger.info(
        "Starting %s experiments",
        total_models,
    )

    for idx, (name, cfg) in enumerate(
        models.items(),
        start=1,
    ):
        logger.info(
            "[%s/%s] Running %s",
            idx,
            total_models,
            name,
        )

        input_type = cfg["input"]
        model = cfg["model"]

        start_time = time.perf_counter()

        logger.info("[%s] Training...", name)

        if input_type == "df":
            model.fit(train_df)

        elif input_type == "matrix":
            model.fit(train_matrix)

        else:
            raise ValueError(f"Unknown input type: {input_type}")

        train_time = time.perf_counter() - start_time

        logger.info(
            "[%s] Training finished in %.2fs",
            name,
            train_time,
        )

        logger.info(
            "[%s] Evaluating...",
            name,
        )

        metrics = evaluate_model_ncf(
            model=model,
            train_df=train_df,
            test_df=test,
            n_items=n_items,
            k=k,
            num_eval_negatives=num_eval_negatives,
            seed=random_seed,
        )

        logger.info(
            "[%s] Hit Rate@%s=%.4f | Precision@%s=%.4f | NDCG@%s=%.4f | " "MRR@%s=%.4f",
            name,
            k,
            metrics["hit_rate@k"],
            k,
            metrics["precision@k"],
            k,
            metrics["ndcg@k"],
            k,
            metrics["mrr@k"],
        )

        results.append(
            {
                "model": name,
                "estimator": model,
                "input": input_type,
                **metrics,
                "train_time_seconds": round(
                    train_time,
                    2,
                ),
            }
        )

    logger.info("All experiments finished")

    results_sorted = sorted(
        results,
        key=lambda x: x["ndcg@k"],
        reverse=True,
    )

    table = Table(title="Experiment Ranking")

    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Hit Rate@K", justify="right")
    table.add_column("Precision@K", justify="right")
    table.add_column("NDCG@K", justify="right")
    table.add_column("MRR@K", justify="right")
    table.add_column("Train Time (s)", justify="right")

    for pos, result in enumerate(
        results_sorted,
        start=1,
    ):
        table.add_row(
            str(pos),
            result["model"],
            f'{result["hit_rate@k"]:.4f}',
            f'{result["precision@k"]:.4f}',
            f'{result["ndcg@k"]:.4f}',
            f'{result["mrr@k"]:.4f}',
            str(result["train_time_seconds"]),
        )

    console.print(table)

    return results
