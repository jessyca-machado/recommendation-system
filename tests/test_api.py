import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest_asyncio.fixture
async def client():
    async with LifespanManager(app):
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            yield client


async def test_healthcheck(client):
    response = await client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"


async def test_recommendations_returns_items(client):
    response = await client.get("/recommend/0")

    assert response.status_code == 200

    data = response.json()

    assert "recommendations" in data

    assert len(data["recommendations"]) == 10


async def test_recommendation_schema(client):
    response = await client.get("/recommend/0")

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    for item in recommendations:
        assert "rank" in item
        assert "score" in item
        assert "itemid" in item


async def test_recommendations_are_ranked(client):
    response = await client.get("/recommend/0")

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    ranks = [item["rank"] for item in recommendations]

    assert ranks == list(range(1, 11))


async def test_custom_k_recommendations(client):
    response = await client.get("/recommend/0?k=5")

    assert response.status_code == 200

    recommendations = response.json()["recommendations"]

    assert len(recommendations) == 5


async def test_healthcheck_model_loaded(client):
    response = await client.get("/health")

    body = response.json()

    assert body["model_loaded"] is True
