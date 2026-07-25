from src.config.params import load_params
from src.models.factory import build_models

from src.models.popularity import PopularityRecommender
from src.models.als import ALSRecommender
from src.models.bpr import BPRRecommender
from src.models.knn import KNNRecommender
from src.models.mlp import MLPRecommender


def test_build_models():
    """
    Valida se o factory cria todos os modelos configurados.
    """

    params = load_params()

    models = build_models(
        params["models"]
    )

    assert set(models.keys()) == {
        "popularity",
        "als",
        "bpr",
        "knn",
        "mlp",
    }


def test_models_instances():
    """
    Valida se cada modelo criado pertence à implementação esperada.
    """

    params = load_params()

    models = build_models(
        params["models"]
    )

    expected_types = {
        "popularity": PopularityRecommender,
        "als": ALSRecommender,
        "bpr": BPRRecommender,
        "knn": KNNRecommender,
        "mlp": MLPRecommender,
    }

    for name, config in models.items():
        assert isinstance(
            config["model"],
            expected_types[name],
        )


def test_models_have_required_config():
    """
    Valida se todos os modelos possuem contrato esperado.
    """

    params = load_params()

    models = build_models(
        params["models"]
    )

    for config in models.values():
        assert "input" in config
        assert "model" in config
        assert config["model"] is not None
