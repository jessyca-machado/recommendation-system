"""API de inferência: serve o modelo registrado no MLflow via FastAPI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.config.settings import settings
from src.inference.catalog import load_item_catalog
from src.inference.loader import load_production_model
from src.inference.recommend import recommend_for_user

_state: dict = {}


def _load_model() -> None:
    """Carrega o modelo promovido para staging."""

    _state["model"] = load_production_model(
        model_name=settings.mlflow_registered_model_name,
        alias="staging",
    )
    _state["item_catalog"] = load_item_catalog()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    _state.clear()


app = FastAPI(
    title="Recommendation System API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "Recommendation System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend/{user_idx}",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in _state,
    }


@app.get("/recommend/{user_idx}")
def recommend(
    user_idx: int,
    k: int = 10,
):
    if "model" not in _state:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded.",
        )

    recommendations = recommend_for_user(
        model=_state["model"],
        item_catalog=_state["item_catalog"],
        user_idx=user_idx,
        k=k,
    )

    return {
        "user_idx": user_idx,
        "recommendations": recommendations.to_dict(
            orient="records",
        ),
    }
