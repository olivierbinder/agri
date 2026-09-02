import pydantic as pdt
from fastapi import Depends, FastAPI

from agri.api import logic
from agri.api.dependencies import get_model
from agri.core import constants
from agri.io import registries

app = FastAPI(title="Agri Yield Prediction API 🌾")


# FastAPI uses Pydantic BaseModel to validate the payload automatically
class PredictRequest(pdt.BaseModel):
    Area: str = constants.DEFAULT_AREA
    Item: str = constants.DEFAULT_ITEM
    Year: int = constants.DEFAULT_YEAR
    average_rain_fall_mm_per_year: float = constants.DEFAULT_RAINFALL
    pesticides_tonnes: float = constants.DEFAULT_PESTICIDES
    avg_temp: float = constants.DEFAULT_TEMP


class PredictResponse(pdt.BaseModel):
    prediction: float
    unit: str = constants.YIELD_UNIT


class RecommendRequest(pdt.BaseModel):
    Area: str = constants.DEFAULT_AREA
    Year: int = constants.DEFAULT_YEAR
    average_rain_fall_mm_per_year: float = constants.DEFAULT_RAINFALL
    pesticides_tonnes: float = constants.DEFAULT_PESTICIDES
    avg_temp: float = constants.DEFAULT_TEMP


class CropRecommendation(pdt.BaseModel):
    Item: str
    prediction: float
    relative_score: float


class RecommendResponse(pdt.BaseModel):
    recommendations: list[CropRecommendation]
    unit: str = constants.YIELD_UNIT


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest, model: registries.Loader.Adapter = Depends(get_model)
):
    pred_value = logic.predict_yield(model, **request.model_dump())
    return PredictResponse(prediction=pred_value)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(
    request: RecommendRequest, model: registries.Loader.Adapter = Depends(get_model)
):
    ranked = logic.recommend_crops(model, **request.model_dump())
    recommendations = [CropRecommendation(**row) for row in ranked.to_dict("records")]
    return RecommendResponse(recommendations=recommendations)


@app.get("/health")
def health():
    return {"status": "ok"}
