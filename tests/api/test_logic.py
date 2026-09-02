# %% IMPORTS

from agri.api import logic
from agri.core import constants, models

# %% LOGIC


def test_predict_yield(model: models.RandomForest) -> None:
    # when
    prediction = logic.predict_yield(
        model,  # ty: ignore[invalid-argument-type]
        Area=constants.DEFAULT_AREA,
        Item=constants.DEFAULT_ITEM,
        Year=constants.DEFAULT_YEAR,
        average_rain_fall_mm_per_year=constants.DEFAULT_RAINFALL,
        pesticides_tonnes=constants.DEFAULT_PESTICIDES,
        avg_temp=constants.DEFAULT_TEMP,
    )
    # then
    assert isinstance(prediction, float), "Prediction should be a plain float!"
    assert prediction >= 0, "Yield prediction should never be negative!"


def test_recommend_crops(model: models.RandomForest) -> None:
    # when
    ranking = logic.recommend_crops(
        model,  # ty: ignore[invalid-argument-type]
        Area=constants.DEFAULT_AREA,
        Year=constants.DEFAULT_YEAR,
        average_rain_fall_mm_per_year=constants.DEFAULT_RAINFALL,
        pesticides_tonnes=constants.DEFAULT_PESTICIDES,
        avg_temp=constants.DEFAULT_TEMP,
    )
    # then
    assert set(ranking.columns) == {"Item", "prediction", "relative_score"}, (
        "Ranking should expose Item, prediction and relative_score columns!"
    )
    assert set(ranking["Item"]) == set(constants.ITEMS), (
        "Ranking should cover every known crop, exactly once!"
    )
    assert (ranking["relative_score"].diff().dropna() <= 0).all(), (
        "Ranking should be sorted by relative_score, descending!"
    )
    assert (ranking["prediction"] >= 0).all(), (
        "Yield predictions should never be negative!"
    )
