# %% IMPORTS

import typing as T

import pytest
from fastapi.testclient import TestClient

from agri.api import server
from agri.api.dependencies import get_model
from agri.core import constants, models

# %% FIXTURES


@pytest.fixture
def client(model: models.RandomForest) -> T.Generator[TestClient, None, None]:
    # bypass the Mlflow registry entirely: serve the already-fitted test model
    server.app.dependency_overrides[get_model] = lambda: model
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


# %% ENDPOINTS


def test_health(client: TestClient) -> None:
    # when
    response = client.get("/health")
    # then
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict(client: TestClient) -> None:
    # given
    payload = {
        "Area": constants.DEFAULT_AREA,
        "Item": constants.DEFAULT_ITEM,
        "Year": constants.DEFAULT_YEAR,
        "average_rain_fall_mm_per_year": constants.DEFAULT_RAINFALL,
        "pesticides_tonnes": constants.DEFAULT_PESTICIDES,
        "avg_temp": constants.DEFAULT_TEMP,
    }
    # when
    response = client.post("/predict", json=payload)
    # then
    assert response.status_code == 200
    data = response.json()
    assert data["unit"] == constants.YIELD_UNIT
    assert data["prediction"] >= 0, "Yield prediction should never be negative!"


def test_predict__invalid_payload(client: TestClient) -> None:
    # given: every field has a default, but Year must still be an int
    # when
    response = client.post("/predict", json={"Year": "not-a-year"})
    # then
    assert response.status_code == 422, "FastAPI should reject a wrongly-typed field!"


def test_recommend(client: TestClient) -> None:
    # given
    payload = {
        "Area": constants.DEFAULT_AREA,
        "Year": constants.DEFAULT_YEAR,
        "average_rain_fall_mm_per_year": constants.DEFAULT_RAINFALL,
        "pesticides_tonnes": constants.DEFAULT_PESTICIDES,
        "avg_temp": constants.DEFAULT_TEMP,
    }
    # when
    response = client.post("/recommend", json=payload)
    # then
    assert response.status_code == 200
    data = response.json()
    assert data["unit"] == constants.YIELD_UNIT
    recommendations = data["recommendations"]
    assert {r["Item"] for r in recommendations} == set(constants.ITEMS), (
        "Recommendations should cover every known crop, exactly once!"
    )
    scores = [r["relative_score"] for r in recommendations]
    assert scores == sorted(scores, reverse=True), (
        "Recommendations should be ranked by relative_score, descending!"
    )
