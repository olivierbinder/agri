"""Prediction and recommendation logic shared by the FastAPI server and the Streamlit app.

Parameter names match schemas.InputsSchema's columns (Area, Item, Year, ...) verbatim,
so callers can pass **request.model_dump() straight through from the FastAPI Pydantic
request models.
"""

import pandas as pd

from agri.core import constants, schemas
from agri.io import registries


def predict_yield(
    model: registries.Loader.Adapter,
    Area: str,
    Item: str,
    Year: int,
    average_rain_fall_mm_per_year: float,
    pesticides_tonnes: float,
    avg_temp: float,
) -> float:
    """Predict the yield (hg/ha) for a single crop/plot context."""
    df = pd.DataFrame(
        [
            {
                "Area": Area,
                "Item": Item,
                "Year": Year,
                "average_rain_fall_mm_per_year": average_rain_fall_mm_per_year,
                "pesticides_tonnes": pesticides_tonnes,
                "avg_temp": avg_temp,
            }
        ]
    )
    validated_df = schemas.InputsSchema.check(df)
    outputs = model.predict(validated_df)
    return float(outputs.iloc[0]["prediction"])


def recommend_crops(
    model: registries.Loader.Adapter,
    Area: str,
    Year: int,
    average_rain_fall_mm_per_year: float,
    pesticides_tonnes: float,
    avg_temp: float,
) -> pd.DataFrame:
    """Rank every known crop by relative yield score for a given plot context.

    The relative score is each crop's predicted yield divided by its own global
    reference yield (see constants.CROP_REF_YIELD), so naturally high-yield crops
    (e.g. potatoes) don't always top the ranking regardless of climate.

    Returns a dataframe with columns: Item, prediction, relative_score — sorted by
    relative_score descending.
    """
    context = {
        "Area": Area,
        "Year": Year,
        "average_rain_fall_mm_per_year": average_rain_fall_mm_per_year,
        "pesticides_tonnes": pesticides_tonnes,
        "avg_temp": avg_temp,
    }
    df = pd.DataFrame([{**context, "Item": item} for item in constants.ITEMS])
    validated_df = schemas.InputsSchema.check(df)
    outputs = model.predict(validated_df)

    ranked = df[["Item"]].assign(prediction=outputs["prediction"].to_numpy())
    ranked["relative_score"] = ranked["prediction"] / ranked["Item"].map(
        constants.CROP_REF_YIELD
    )
    return ranked.sort_values("relative_score", ascending=False).reset_index(drop=True)
