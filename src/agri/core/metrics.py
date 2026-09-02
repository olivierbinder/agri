"""Evaluate model performances with metrics."""

# %% IMPORTS

from __future__ import annotations

import abc
import typing as T

import mlflow
import pandas as pd
import pydantic as pdt
from mlflow.metrics import MetricValue
from sklearn import metrics as sklearn_metrics

from agri.core import models, schemas

# %% TYPINGS

# kept as a T.TypeAlias (not `type`) because MlflowMetric(...) is called as a
# constructor at runtime; PEP 695 `type` aliases aren't callable.
MlflowMetric: T.TypeAlias = MetricValue  # noqa: UP040

# %% METRICS


class Metric(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a project metric.

    Use metrics to evaluate model performance.
    e.g., accuracy, precision, recall, MAE, F1, ...

    Parameters:
        name (str): name of the metric for the reporting.
        greater_is_better (bool): maximize or minimize result.
    """

    KIND: str

    name: str
    greater_is_better: bool

    @abc.abstractmethod
    def score(self, targets: schemas.Targets, outputs: schemas.Outputs) -> float:
        """Score the outputs against the targets.

        Args:
            targets (schemas.Targets): expected values.
            outputs (schemas.Outputs): predicted values.

        Returns:
            float: single result from the metric computation.
        """

    def scorer(
        self, model: models.Model, inputs: schemas.Inputs, targets: schemas.Targets
    ) -> float:
        """Score model outputs against targets.

        Args:
            model (models.Model): model to evaluate.
            inputs (schemas.Inputs): model inputs values.
            targets (schemas.Targets): model expected values.

        Returns:
            float: single result from the metric computation.
        """
        outputs = model.predict(inputs=inputs)
        score = self.score(targets=targets, outputs=outputs)
        sign = 1 if self.greater_is_better else -1
        return score * sign

    def to_mlflow(self, suffix: str = "") -> MlflowMetric:
        """Convert the metric to an Mlflow metric.

        Args:
            suffix (str): optional suffix to append to the metric name.

        Returns:
            MlflowMetric: the Mlflow metric.
        """
        metric_name = f"{self.name}{suffix}"

        def eval_fn(
            predictions: pd.Series[int],  # ty: ignore
            targets: pd.Series[int],  # ty: ignore
        ) -> MlflowMetric:
            """Evaluation function associated with the mlflow metric.

            Args:
                predictions (pd.Series): model predictions.
                targets (pd.Series | None): model targets.

            Returns:
                MlflowMetric: the mlflow metric.
            """
            score_targets = schemas.Targets(
                {"hg/ha_yield": targets}, index=targets.index
            )
            score_outputs = schemas.Outputs(
                {"prediction": predictions}, index=predictions.index
            )
            score = self.score(targets=score_targets, outputs=score_outputs)
            return MlflowMetric(aggregate_results={metric_name: score})

        return mlflow.metrics.make_metric(
            eval_fn=eval_fn, name=metric_name, greater_is_better=self.greater_is_better
        )


class SklearnMetric(Metric):
    """Compute metrics with sklearn.

    Parameters:
        name (str): display name for the metric.
        sklearn_name (str | None): name of the sklearn metric function. Defaults to name if None.
        greater_is_better (bool): maximize or minimize.
    """

    KIND: T.Literal["SklearnMetric"] = "SklearnMetric"

    name: str = "mean_squared_error"
    sklearn_name: str | None = None
    greater_is_better: bool = False

    @T.override
    def score(self, targets: schemas.Targets, outputs: schemas.Outputs) -> float:
        func_name = self.sklearn_name if self.sklearn_name else self.name
        metric = getattr(sklearn_metrics, func_name)
        y_true = targets["hg/ha_yield"]
        y_pred = outputs["prediction"]
        score = metric(y_pred=y_pred, y_true=y_true)
        return float(score)


MetricKind = SklearnMetric
type MetricsKind = list[T.Annotated[MetricKind, pdt.Field(discriminator="KIND")]]

# %% THRESHOLDS


class Threshold(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """A project threshold for a metric.

    Use thresholds to monitor model performances.
    e.g., to trigger an alert when a threshold is met.

    Parameters:
        threshold (int | float): absolute threshold value.
        greater_is_better (bool): maximize or minimize result.
    """

    threshold: int | float
    greater_is_better: bool
