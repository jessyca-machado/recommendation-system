from src.inference.catalog import load_item_catalog
from src.inference.loader import load_production_model

MODEL_NAME = "recommendation-model"


def test_load_staging_model():
    """
    Valida que o modelo promovido no MLflow
    pode ser carregado.
    """

    model = load_production_model(
        model_name=MODEL_NAME,
        alias="staging",
    )

    assert model is not None


def test_model_has_recommendation_method():
    """
    Valida que o modelo carregado possui
    interface esperada de recomendação.
    """

    model = load_production_model(
        model_name=MODEL_NAME,
        alias="staging",
    )

    assert hasattr(
        model,
        "recommend",
    )


def test_catalog_loaded():
    catalog = load_item_catalog()

    assert len(catalog) > 0


def test_catalog_has_required_columns():
    catalog = load_item_catalog()

    assert "item_idx" in catalog.columns
    assert "itemid" in catalog.columns
