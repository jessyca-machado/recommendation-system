"""API de inferência: serve o modelo registrado no MLflow via FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.config.settings import settings
from src.inference.catalog import load_item_catalog
from src.inference.loader import load_production_model
from src.inference.recommend import recommend_for_user

_state: dict[str, object] = {}


def _load_model() -> None:
    """Carrega o modelo promovido para produção na API.

    Returns:
        None: Atualiza o estado interno com o modelo e o catálogo.
    """

    _state["model"] = load_production_model(
        model_name=settings.mlflow_registered_model_name,
        alias="production",
    )
    _state["item_catalog"] = load_item_catalog()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Gerencia o ciclo de vida da aplicação FastAPI.

    Args:
        app: Instância da aplicação FastAPI.

    Yields:
        None: Controla o carregamento e descarregamento do modelo.
    """

    _load_model()
    yield
    _state.clear()


app = FastAPI(
    title="Recommendation System API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, object]:
    """Retorna a descrição básica dos endpoints da API.

    Returns:
        dict[str, object]: Informações gerais do serviço.
    """

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
def health() -> dict[str, object]:
    """Retorna o estado de saúde da API.

    Returns:
        dict[str, object]: Indicador de carregamento do modelo.
    """

    return {
        "status": "ok",
        "model_loaded": "model" in _state,
    }


@app.get("/recommend/{user_idx}")
def recommend(
    user_idx: int,
    k: int = 10,
) -> dict[str, object]:
    """Gera recomendações para um usuário específico.

    Args:
        user_idx: Identificador do usuário.
        k: Quantidade de recomendações desejadas.

    Returns:
        dict[str, object]: Payload com as recomendações do usuário.

    Raises:
        HTTPException: Se o modelo não estiver carregado.
    """

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
