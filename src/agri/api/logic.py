"""Prediction and recommendation logic used by the FastAPI server.

Parameter names match schemas.InputsSchema's columns (Area, Item, Year, ...) verbatim,
so callers can pass **request.model_dump() straight through from the FastAPI Pydantic
request models.
"""

import pandas as pd

from agri.core import constants, schemas
from agri.io import registries

# Absolute tolerances used to decide whether the user's inputs match the real
# conditions recorded for a held-out year. Above these, the row's actual yield
# describes a different plot, so comparing prediction to it would be meaningless.
ACTUAL_RAINFALL_TOLERANCE = 10.0
ACTUAL_PESTICIDES_TOLERANCE = 10.0
ACTUAL_TEMP_TOLERANCE = 0.5

# Columns `actuals` must carry for a comparison to be possible at all.
ACTUAL_FEATURES = (
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
)
ACTUAL_COLUMNS = ("Area", "Item", "Year", *ACTUAL_FEATURES, "hg/ha_yield")


def has_actual_conditions(actuals: pd.DataFrame) -> bool:
    """Return True when `actuals` carries the condition columns needed to compare.

    A reference file with only (Area, Item, Year, yield) cannot tell whether a prediction
    was made for the same plot as the measured yield, so comparisons must be refused
    rather than silently wrong.
    """
    return set(ACTUAL_COLUMNS).issubset(actuals.columns)


# %% LOOKUPS


def lookup_actual_row(
    actuals: pd.DataFrame, Area: str, Item: str, Year: int
) -> dict | None:
    """Return the recorded real conditions and yield for Area/Item/Year, or None.

    `actuals` only ever has one year (2013, the model's held-out test year) — this
    naturally returns None for any other Year, which is the point: it's "extrapolation,
    no ground truth" rather than "an actual value we chose not to show".
    """
    if not has_actual_conditions(actuals):
        return None
    match = actuals[
        (actuals["Area"] == Area)
        & (actuals["Item"] == Item)
        & (actuals["Year"] == Year)
    ]
    return match.iloc[0].to_dict() if not match.empty else None


def lookup_actual_yield(
    actuals: pd.DataFrame,
    Area: str,
    Item: str,
    Year: int,
    average_rain_fall_mm_per_year: float,
    pesticides_tonnes: float,
    avg_temp: float,
) -> float | None:
    """Return the real yield (hg/ha) for Area/Item/Year, or None if not comparable.

    The actual yield is only meaningful when the user's climate/pesticide inputs match
    the real conditions recorded for that year; otherwise we'd be comparing a prediction
    made for other conditions against a yield measured under different ones. Any
    mismatch (or a year without ground truth) returns None.
    """
    row = lookup_actual_row(actuals, Area=Area, Item=Item, Year=Year)
    if row is None:
        return None
    rainfall, pesticides, temp = (float(row[column]) for column in ACTUAL_FEATURES)
    if (
        abs(average_rain_fall_mm_per_year - rainfall) > ACTUAL_RAINFALL_TOLERANCE
        or abs(pesticides_tonnes - pesticides) > ACTUAL_PESTICIDES_TOLERANCE
        or abs(avg_temp - temp) > ACTUAL_TEMP_TOLERANCE
    ):
        return None
    return float(row["hg/ha_yield"])


def conditions_match(actuals: pd.DataFrame, Area: str, Year: int) -> bool:
    """Return True when `actuals` records usable conditions for Area/Year."""
    return (
        has_actual_conditions(actuals)
        and not actuals[(actuals["Area"] == Area) & (actuals["Year"] == Year)].empty
    )


# %% PREDICTION


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
    actuals: pd.DataFrame,
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

    Returns a dataframe with columns: Item, prediction, relative_score, actual —
    sorted by relative_score descending. `actual` is None wherever `actuals` has
    no matching row (see lookup_actual_yield).
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
    ranked["actual"] = [
        lookup_actual_yield(
            actuals,
            Area=Area,
            Item=item,
            Year=Year,
            average_rain_fall_mm_per_year=average_rain_fall_mm_per_year,
            pesticides_tonnes=pesticides_tonnes,
            avg_temp=avg_temp,
        )
        for item in ranked["Item"]
    ]
    return ranked.sort_values("relative_score", ascending=False).reset_index(drop=True)
