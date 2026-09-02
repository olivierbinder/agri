import pandas as pd
import pydantic as pdt
from fastapi import Depends, FastAPI

from agri.api.dependencies import get_model
from agri.core import constants, schemas
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
    # 1. Convert the pydantic payload into a pandas DataFrame (1 row)
    df = pd.DataFrame([request.model_dump()])

    # 2. Let Pandera validate it against your robust core schema
    validated_df = schemas.InputsSchema.check(df)

    # 3. Call the model
    outputs = model.predict(validated_df)

    # 4. Extract the prediction value (outputs is also a DataFrame)
    pred_value = float(outputs.iloc[0]["prediction"])

    return PredictResponse(prediction=pred_value)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(
    request: RecommendRequest, model: registries.Loader.Adapter = Depends(get_model)
):
    # 1. Build one row per known crop, sharing the same plot context
    context = request.model_dump()
    df = pd.DataFrame([{**context, "Item": item} for item in constants.ITEMS])

    # 2. Let Pandera validate it against the core schema
    validated_df = schemas.InputsSchema.check(df)

    # 3. Call the model once for every crop
    outputs = model.predict(validated_df)

    # 4. Normalize each crop's prediction by its own global reference yield, so
    #    naturally high-yield crops (e.g. Potatoes) don't always top the ranking
    #    regardless of climate. Rank by that relative score, descending.
    ranked = df[["Item"]].assign(prediction=outputs["prediction"].to_numpy())
    ranked["relative_score"] = ranked["prediction"] / ranked["Item"].map(
        constants.CROP_REF_YIELD
    )
    ranked = ranked.sort_values("relative_score", ascending=False)

    recommendations = [
        CropRecommendation(
            Item=row.Item,
            prediction=float(row.prediction),
            relative_score=float(row.relative_score),
        )
        for row in ranked.itertuples()
    ]

    return RecommendResponse(recommendations=recommendations)


@app.get("/health")
def health():
    return {"status": "ok"}
