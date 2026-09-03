"""Define trainable machine learning models."""

# %% IMPORTS

import abc
import typing as T

import pandas as pd
import pydantic as pdt
import shap
from sklearn import ensemble, pipeline

from agri.core import features, schemas

# %% TYPES

# Model params.
ParamKey = str
ParamValue = T.Any
Params = dict[ParamKey, ParamValue]

# %% MODELS - BASE CLASS


from sklearn.base import BaseEstimator


class Model(
    abc.ABC, pdt.BaseModel, BaseEstimator, strict=True, frozen=False, extra="forbid"
):  # ty: ignore
    """Base class for a project model.

    Use a model to adapt AI/ML frameworks.
    e.g., to swap easily one model with another.
    """

    KIND: str

    def get_params(self, deep: bool = True) -> Params:
        """Get the model params.

        Args:
            deep (bool, optional): ignored.

        Returns:
            Params: internal model parameters.
        """
        params: Params = {}
        for key in self.model_fields:
            if not key.startswith("_") and not key.isupper():
                params[key] = getattr(self, key)
        return params

    def set_params(self, **params: ParamValue) -> T.Self:
        """Set the model params in place.

        Returns:
            T.Self: instance of the model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abc.abstractmethod
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> T.Self:
        """Fit the model on the given inputs and targets.

        Args:
            inputs (schemas.Inputs): model training inputs.
            targets (schemas.Targets): model training targets.

        Returns:
            T.Self: instance of the model.
        """

    @abc.abstractmethod
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        """Generate outputs with the model for the given inputs.

        Args:
            inputs (schemas.Inputs): model prediction inputs.

        Returns:
            schemas.Outputs: model prediction outputs.
        """

    def explain_model(self) -> schemas.FeatureImportances:
        """Explain the internal model structure.

        Returns:
            schemas.FeatureImportances: feature importances.
        """
        raise NotImplementedError()

    def explain_samples(self, inputs: schemas.Inputs) -> schemas.SHAPValues:
        """Explain model outputs on input samples.

        Returns:
            schemas.SHAPValues: SHAP values.
        """
        raise NotImplementedError()

    def get_internal_model(self) -> T.Any:
        """Return the internal model in the object.

        Raises:
            NotImplementedError: method not implemented.

        Returns:
            T.Any: any internal model (either empty or fitted).
        """
        raise NotImplementedError()


# %% MODELS - RANDOM FOREST


class RandomForest(Model):
    """Simple baseline model based on scikit-learn.

    Parameters:
        max_depth (int): maximum depth of the random forest.
        n_estimators (int): number of estimators in the random forest.
        random_state (int, optional): random state of the machine learning pipeline.
    """

    KIND: T.Literal["RandomForest"] = "RandomForest"

    # params
    max_depth: int = 20
    n_estimators: int = 200
    random_state: int | None = 42
    preprocessor: features.PreprocessorKind = pdt.Field(
        default_factory=features.AgriPreprocessor, discriminator="KIND"
    )
    # private
    _pipeline: pipeline.Pipeline | None = None

    @T.override
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> "RandomForest":
        transformer = self.preprocessor.build_transformer(
            random_state=self.random_state
        )
        regressor = ensemble.RandomForestRegressor(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        # pipeline
        self._pipeline = pipeline.Pipeline(
            steps=[
                ("preprocessor", transformer),
                ("regressor", regressor),
            ]
        )
        self._pipeline.fit(X=inputs, y=targets["hg/ha_yield"])
        return self

    @T.override
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        model = self.get_internal_model()
        prediction = model.predict(inputs)
        outputs_ = pd.DataFrame(
            data={schemas.OutputsSchema.prediction: prediction}, index=inputs.index
        )
        outputs = schemas.OutputsSchema.check(data=outputs_)
        return outputs

    @T.override
    def explain_model(self) -> schemas.FeatureImportances:
        model = self.get_internal_model()
        regressor = model.named_steps["regressor"]
        # transformer is all steps before regressor
        transformer = model[:-1]
        feature = transformer.get_feature_names_out()
        feature_importances_ = pd.DataFrame(
            data={
                "feature": feature,
                "importance": regressor.feature_importances_,
            }
        )
        feature_importances = schemas.FeatureImportancesSchema.check(
            data=feature_importances_
        )
        return feature_importances

    @T.override
    def explain_samples(self, inputs: schemas.Inputs) -> schemas.SHAPValues:
        model = self.get_internal_model()
        regressor = model.named_steps["regressor"]
        transformer = model[:-1]
        transformed = transformer.transform(X=inputs)
        explainer = shap.TreeExplainer(model=regressor)
        shap_values_ = pd.DataFrame(
            data=explainer.shap_values(X=transformed),
            columns=transformer.get_feature_names_out(),
        )
        shap_values = schemas.SHAPValuesSchema.check(data=shap_values_)
        return shap_values

    @T.override
    def get_internal_model(self) -> pipeline.Pipeline:
        model = self._pipeline
        if model is None:
            raise ValueError("Model is not fitted yet!")
        return model


# %% MODELS - XGBOOST
import xgboost as xgb


class XGBoost(Model):
    """Advanced model based on XGBoost.

    Parameters:
        max_depth (int): maximum depth of the xgboost.
        n_estimators (int): number of estimators.
        learning_rate (float): learning rate.
        subsample (float): subsample.
        colsample_bytree (float): colsample_bytree.
        random_state (int, optional): random state.
    """

    KIND: T.Literal["XGBoost"] = "XGBoost"

    # params
    max_depth: int = 6
    n_estimators: int = 300
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int | None = 42
    preprocessor: features.PreprocessorKind = pdt.Field(
        default_factory=features.AgriPreprocessor, discriminator="KIND"
    )

    # private
    _pipeline: pipeline.Pipeline | None = None

    @T.override
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> "XGBoost":
        transformer = self.preprocessor.build_transformer(
            random_state=self.random_state
        )
        regressor = xgb.XGBRegressor(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._pipeline = pipeline.Pipeline(
            steps=[
                ("preprocessor", transformer),
                ("regressor", regressor),
            ]
        )
        self._pipeline.fit(X=inputs, y=targets["hg/ha_yield"])
        return self

    @T.override
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        model = self.get_internal_model()
        prediction = model.predict(inputs)
        # XGBoost can sometimes output negative numbers which violates schemas
        prediction = prediction.clip(min=0)
        outputs_ = pd.DataFrame(
            data={schemas.OutputsSchema.prediction: prediction}, index=inputs.index
        )
        outputs = schemas.OutputsSchema.check(data=outputs_)
        return outputs

    @T.override
    def explain_model(self) -> schemas.FeatureImportances:
        model = self.get_internal_model()
        regressor = model.named_steps["regressor"]
        transformer = model[:-1]
        feature = transformer.get_feature_names_out()
        feature_importances_ = pd.DataFrame(
            data={
                "feature": feature,
                "importance": regressor.feature_importances_,
            }
        )
        feature_importances = schemas.FeatureImportancesSchema.check(
            data=feature_importances_
        )
        return feature_importances

    @T.override
    def explain_samples(self, inputs: schemas.Inputs) -> schemas.SHAPValues:
        model = self.get_internal_model()
        regressor = model.named_steps["regressor"]
        transformer = model[:-1]
        transformed = transformer.transform(X=inputs)
        explainer = shap.TreeExplainer(model=regressor)
        shap_values_ = pd.DataFrame(
            data=explainer.shap_values(X=transformed),
            columns=transformer.get_feature_names_out(),
        )
        shap_values = schemas.SHAPValuesSchema.check(data=shap_values_)
        return shap_values

    @T.override
    def get_internal_model(self) -> pipeline.Pipeline:
        model = self._pipeline
        if model is None:
            raise ValueError("Model is not fitted yet!")
        return model


ModelKind = RandomForest | XGBoost
