"""Runner de experimentos: treina várias configurações, registra cada uma no
MLflow e promove a melhor (por NDCG) a produção no Model Registry.

Uso:
    python src/experiments/run_all.py
"""
import json
import logging

import mlflow
from rich.console import Console
from rich.logging import RichHandler
from scipy.sparse import csr_matrix

from src.config.params import load_params
from src.config.settings import configure_runtime, settings
from src.experiments.runner import run_experiments
from src.models.factory import build_models
from src.prepare import (
    apply_mappings,
    build_mappings,
    load_data,
    load_train_test,
    save_datasets,
    split_train_test,
)

configure_runtime(settings)

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
    params = load_params()

    models = build_models(params["models"])

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="experiment_runner"):
        mlflow.log_params(
            {
                "tracking_uri": settings.mlflow_tracking_uri,
                "experiment": settings.mlflow_experiment_name,
                **params["experiment"],
                **params["weights"],
            }
        )

        logger.info("Loading dataset...")
        df = load_data()

        logger.info("Splitting train/test...")
        train, test = split_train_test(df)

        logger.info("Building mappings...")
        train, user2idx, item2idx = build_mappings(train)
        test = apply_mappings(test, user2idx, item2idx)

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

        train["weight"] = train["event"].map(params["weights"])

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

        mlflow.log_params(
            {
                "matrix_rows": train_matrix.shape[0],
                "matrix_cols": train_matrix.shape[1],
                "num_users": len(user2idx),
                "num_items": len(item2idx),
            }
        )

        logger.info("Running experiments...")

        results = run_experiments(
            train_df=train,
            train_matrix=train_matrix,
            test=test,
            models=models,
            k=params["experiment"]["top_k"],
            n_items=len(item2idx),
            num_eval_negatives=params["experiment"]["num_eval_negatives"],
            random_seed=params["experiment"]["random_seed"],
        )

        logger.info("Finished %s experiments", len(results))

        mlflow.log_metric(
            "num_experiments",
            len(results),
        )

        logger.info("Saving metrics...")

        settings.metrics_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        settings.models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_file = settings.metrics_dir / "experiment_results.json"

        with metrics_file.open("w") as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact(str(metrics_file))

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

        mlflow.log_param(
            "best_model",
            best_model["model"],
        )

        metric_mapping = {
            "ndcg@k": "best_ndcg_at_10",
            "hit_rate@k": "best_hit_rate_at_10",
            "precision@k": "best_precision_at_10",
            "recall@k": "best_recall_at_10",
            "mrr@k": "best_mrr_at_10",
        }

        for source_metric, target_metric in metric_mapping.items():
            if source_metric in best_model:
                mlflow.log_metric(
                    target_metric,
                    best_model[source_metric],
                )

        logger.info("Saving best model metadata...")

        best_model_file = settings.models_dir / "best_model.json"

        with best_model_file.open("w") as f:
            json.dump(best_model, f, indent=2)

        mlflow.log_artifact(str(best_model_file))

        logger.info("Pipeline finished successfully ✅")


if __name__ == "__main__":
    main()
