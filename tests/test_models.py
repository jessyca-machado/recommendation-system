from src.config.params import load_params
from src.models.factory import build_models
from src.models.base import RecommenderBase


def test_all_models_inherit_base():

    params = load_params()

    models = build_models(params["models"])

    for name, cfg in models.items():

        assert isinstance(
            cfg["model"],
            RecommenderBase,
        )


def test_all_models_have_input_type():

    params = load_params()

    models = build_models(params["models"])

    for name, cfg in models.items():

        assert cfg["input"] in (
            "df",
            "matrix",
        )


def test_all_models_are_enabled():

    params = load_params()

    models = build_models(params["models"])

    enabled = {
        name
        for name, cfg in params["models"].items()
        if cfg["enabled"]
    }

    assert set(models.keys()) == enabled
