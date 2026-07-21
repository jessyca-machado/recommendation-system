
"""Runner de experimentos: treina várias configurações, registra cada uma no
MLflow e promove a melhor (por NDCG) a produção no Model Registry.

Uso:
    python src/experiments/run_all.py
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import logging

import mlflow
from mlflow import MlflowClient

from rich.console import Console
from rich.logging import RichHandler

from scipy.sparse import csr_matrix

from runner import run_experiments
from src.prepare import (
    build_mappings,
    load_data,
    load_train_test,
    save_datasets,
    split_train_test,
    apply_mappings,
)

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
        )
    ],
)

logger = logging.getLogger("experiments")


def main():

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("recommendation-system")
    client = MlflowClient()

    with mlflow.start_run(run_name="experiment_runner"):

        logger.info("Loading dataset...")
        df = load_data()

        logger.info("Splitting train/test...")
        train, test = split_train_test(df)

        logger.info("Building mappings...")
        train, user2idx, item2idx = build_mappings(train)
        test = apply_mappings(
            test,
            user2idx,
            item2idx
        )

        logger.info(
            "Users: %s | Items: %s",
            len(user2idx),
            len(item2idx),
        )

        logger.info("Saving datasets...")
        save_datasets(
            train=train,
            test=test,
        )

        logger.info("Reloading datasets...")
        train, test = load_train_test()

        train["weight"] = train["event"].map(
            {
                "view": 1,
                "addtocart": 3,
                "transaction": 5,
            }
        )

        logger.info("Creating sparse interaction matrix...")

        train_matrix = csr_matrix(
            (
                train["weight"],
                (train["user_idx"], train["item_idx"]),
            ),
            dtype="float32",
        )

        train_matrix = train_matrix.tocsr()

        logger.info(
            "Matrix shape: %s x %s",
            train_matrix.shape[0],
            train_matrix.shape[1],
        )

        logger.info(
            "Matrix type: %s",
            type(train_matrix).__name__,
        )

        mlflow.log_param("matrix_rows", train_matrix.shape[0],)

        mlflow.log_param("matrix_cols", train_matrix.shape[1],)

        logger.info("Running experiments...")

        results = run_experiments(
            train_df=train,
            train_matrix=train_matrix,
            test=test,
            k=10,
            n_items=len(item2idx),
            num_eval_negatives=99,
        )

        logger.info("Finished %s experiments", len(results))

        mlflow.log_metric("num_experiments", len(results),)

        logger.info("Saving metrics...")

        os.makedirs("metrics", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        with open("metrics/experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact(
            "metrics/experiment_results.json"
        )

        logger.info("Selecting best model...")

        best_model = max(
            results,
            key=lambda x: x["ndcg@k"],
        )

        logger.info(
            "Best model: %s | NDCG@K: %.5f",
            best_model["model"],
            best_model["ndcg@k"],
        )

        mlflow.log_param("best_model", best_model["model"],)

        mlflow.log_metric("best_ndcg_at_10",best_model["ndcg@k"],)

        if "precision@k" in best_model:
            mlflow.log_metric("best_precision_at_10", best_model["precision@k"],)

        if "recall@k" in best_model:
            mlflow.log_metric("best_recall_at_10", best_model["recall@k"],)

        if "mrr@k" in best_model:
            mlflow.log_metric("best_mrr_at_10", best_model["mrr@k"],)

        logger.info("Saving best model metadata...")

        with open("models/best_model.json", "w") as f:
            json.dump(best_model, f, indent=2)

        mlflow.log_artifact("models/best_model.json")

        logger.info("Pipeline finished successfully ✅")


if __name__ == "__main__":
    main()
