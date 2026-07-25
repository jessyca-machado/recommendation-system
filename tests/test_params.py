from src.config.params import load_params


def test_params_load():

    params = load_params()

    assert "experiment" in params
    assert "models" in params
    assert "weights" in params

    assert params["experiment"]["top_k"] > 0
    assert params["experiment"]["random_seed"] == 42
