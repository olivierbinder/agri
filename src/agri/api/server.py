import pandas as pd
import pydantic as pdt
from fastapi import Depends, FastAPI

from agri.api import logic
from agri.api.dependencies import get_actuals, get_model
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
    actual: float | None = None
    unit: str = constants.YIELD_UNIT
    # Real conditions recorded for the requested Area/Item/Year, when available. The UI
    # uses them to pre-fill the sliders and to explain why `actual` is None.
    reference_conditions: dict[str, float] | None = None


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
    actual: float | None = None


class RecommendResponse(pdt.BaseModel):
    recommendations: list[CropRecommendation]
    unit: str = constants.YIELD_UNIT


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    model: registries.Loader.Adapter = Depends(get_model),
    actuals: pd.DataFrame = Depends(get_actuals),
):
    pred_value = logic.predict_yield(model, **request.model_dump())
    actual = logic.lookup_actual_yield(actuals, **request.model_dump())
    row = logic.lookup_actual_row(
        actuals, Area=request.Area, Item=request.Item, Year=request.Year
    )
    conditions = (
        {feature: float(row[feature]) for feature in logic.ACTUAL_FEATURES}
        if row is not None
        else None
    )
    return PredictResponse(
        prediction=pred_value, actual=actual, reference_conditions=conditions
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(
    request: RecommendRequest,
    model: registries.Loader.Adapter = Depends(get_model),
    actuals: pd.DataFrame = Depends(get_actuals),
):
    ranked = logic.recommend_crops(model, actuals, **request.model_dump())
    recommendations = [CropRecommendation(**row) for row in ranked.to_dict("records")]
    return RecommendResponse(recommendations=recommendations)


@app.get("/health")
def health():
    return {"status": "ok"}
