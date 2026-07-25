from unittest.mock import MagicMock, patch

from src.experiments.runner import run_experiments


@patch("src.experiments.runner.evaluate_model_ncf")
def test_runner_returns_results(mock_evaluate):

    mock_evaluate.return_value = {
        "hit_rate@k": 0.80,
        "precision@k": 0.50,
        "ndcg@k": 0.60,
        "mrr@k": 0.70,
    }

    model = MagicMock()

    models = {
        "dummy": {
            "model": model,
            "input": "df",
        }
    }

    results = run_experiments(
        train_df=MagicMock(),
        train_matrix=MagicMock(),
        test=MagicMock(),
        models=models,
        k=10,
        n_items=100,
        num_eval_negatives=99,
    )

    model.fit.assert_called_once()

    assert len(results) == 1

    assert results[0]["model"] == "dummy"

    assert results[0]["ndcg@k"] == 0.60
