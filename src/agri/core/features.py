"""Define custom feature engineering transformers."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from agri.core.constants import CROP_OPT_TEMPS


class AgriFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering for agricultural data."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AgriFeatureEngineer":
        self.feature_names_in_ = np.array(X.columns)
        self.feature_names_out_ = np.array(
            list(X.columns)
            + [
                "rain_temp_interaction",
                "rain_efficiency",
                "temp_deviation",
            ]
        )
        return self

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # 1. Interactions
        X_out["rain_temp_interaction"] = (
            X_out["average_rain_fall_mm_per_year"] * X_out["avg_temp"]
        )
        X_out["rain_efficiency"] = X_out["average_rain_fall_mm_per_year"] / (
            X_out["avg_temp"] + 1
        )

        # 2. Temperature deviation
        opt_temps = X_out["Item"].map(CROP_OPT_TEMPS).fillna(22)
        X_out["temp_deviation"] = (X_out["avg_temp"] - opt_temps).abs()

        return X_out


import abc
import typing as T

import pydantic as pdt
from sklearn import compose, preprocessing


class Preprocessor(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a data preprocessor."""

    KIND: str

    @abc.abstractmethod
    def build_transformer(self, random_state: int | None = None) -> TransformerMixin:
        """Build and return the scikit-learn transformer pipeline."""


class AgriPreprocessor(Preprocessor):
    """Agricultural data preprocessor."""

    KIND: T.Literal["AgriPreprocessor"] = "AgriPreprocessor"

    categoricals: list[str] = ["Area", "Item"]
    numericals: list[str] = [
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
        "rain_temp_interaction",
        "rain_efficiency",
        "temp_deviation",
    ]

    @T.override
    def build_transformer(self, random_state: int | None = None) -> TransformerMixin:
        from sklearn.model_selection import KFold
        from sklearn.pipeline import Pipeline

        categoricals_transformer = preprocessing.TargetEncoder(
            target_type="continuous",
            cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
        )

        transformer = compose.ColumnTransformer(
            [
                ("categoricals", categoricals_transformer, self.categoricals),
                ("numericals", "passthrough", self.numericals),
            ],
            remainder="drop",
        )

        return Pipeline(
            steps=[
                ("feature_engineer", AgriFeatureEngineer()),
                ("transformer", transformer),
            ]
        )  # ty: ignore


PreprocessorKind = AgriPreprocessor
