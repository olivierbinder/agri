# %% IMPORTS

import mlflow
import pandas as pd
import pytest

from agri.core import metrics, models, schemas

# %% METRICS


@pytest.mark.parametrize(
    "name, sklearn_name, interval, greater_is_better",
    [
        ("MSE", "mean_squared_error", [0, float("inf")], False),
        ("MAE", "mean_absolute_error", [0, float("inf")], False),
        ("R2", "r2_score", [float("-inf"), 1], True),
    ],
)
def test_sklearn_metric(
    name: str,
    sklearn_name: str,
    interval: tuple[float, float],
    greater_is_better: bool,
    model: models.RandomForest,
    inputs: schemas.Inputs,
    targets: schemas.Targets,
    outputs: schemas.Outputs,
) -> None:
    # given
    low, high = interval
    data = pd.concat([targets, outputs], axis="columns")
    metric = metrics.SklearnMetric(
        name=name, sklearn_name=sklearn_name, greater_is_better=greater_is_better
    )
    # when
    score = metric.score(targets=targets, outputs=outputs)
    scorer = metric.scorer(model=model, inputs=inputs, targets=targets)
    mlflow_metric = metric.to_mlflow()
    mlflow_results = mlflow.evaluate(
        data=data,
        predictions="prediction",
        targets="hg/ha_yield",
        extra_metrics=[mlflow_metric],
    )
    # then
    # - score
    assert low <= score <= high, "Score should be in the expected interval!"
    # - scorer
    sign = 1 if greater_is_better else -1
    assert scorer == pytest.approx(score * sign), "Scorer should be the signed score!"
    # - mlflow metric
    assert mlflow_metric.name == metric.name, (  # ty: ignore[unresolved-attribute]
        "Mlflow metric name should be the same!"
    )
    assert mlflow_metric.greater_is_better == metric.greater_is_better, (  # ty: ignore[unresolved-attribute]
        "Mlflow metric greater is better should be the same!"
    )
    # - mlflow results
    assert mlflow_results.metrics[metric.name] == pytest.approx(score), (
        "Mlflow results metric should match the direct score!"
    )


# %% THRESHOLDS


def test_threshold() -> None:
    # given
    threshold = metrics.Threshold(threshold=10, greater_is_better=True)
    # then
    assert threshold.threshold == 10, "Threshold value should be set!"
    assert threshold.greater_is_better is True, "Greater is better should be set!"
