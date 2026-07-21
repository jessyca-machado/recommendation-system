# src/experiments/runner.py

import logging
import time

from rich.console import Console
from rich.table import Table

from registry import MODEL_REGISTRY
from src.experiments.evaluator import evaluate_model_ncf

logger = logging.getLogger("runner")
console = Console()


def run_experiments(train_df, train_matrix, test, k=10, n_items=None, num_eval_negatives=99):

    results = []

    total_models = len(MODEL_REGISTRY)

    logger.info(
        "Starting %s experiments",
        total_models,
    )

    for idx, (name, cfg) in enumerate(
        MODEL_REGISTRY.items(),
        start=1,
    ):

        logger.info(
            "[%s/%s] Running %s",
            idx,
            total_models,
            name,
        )

        ModelClass = cfg["model"]
        input_type = cfg["input"]

        model = ModelClass()

        start_time = time.perf_counter()

        logger.info("[%s] Training...", name)

        if input_type == "df":
            model.fit(train_df)

        elif input_type == "matrix":
            model.fit(train_matrix)

        else:
            raise ValueError(
                f"Unknown input type: {input_type}"
            )

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
            seed=42,
        )

        logger.info(
            "[%s] Hit Rate@%s=%.4f | Precision@%s=%.4f | NDCG@%s=%.4f | MRR@%s=%.4f",
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
