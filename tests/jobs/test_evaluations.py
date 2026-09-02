# %% IMPORTS

import pytest

from agri import jobs
from agri.core import metrics
from agri.io import datasets, registries, services

# %% JOBS


@pytest.mark.parametrize(
    "thresholds, should_fail",
    [
        ({"R2_test": metrics.Threshold(threshold=-100, greater_is_better=True)}, False),
        pytest.param(
            {"R2_test": metrics.Threshold(threshold=100, greater_is_better=True)},
            True,
            marks=pytest.mark.xfail(
                reason="R2 can never reach 100.", raises=ValueError
            ),
        ),
    ],
)
def test_evaluations_job(
    thresholds: dict[str, metrics.Threshold],
    should_fail: bool,
    mlflow_service: services.MlflowService,
    logger_service: services.LoggerService,
    inputs_reader: datasets.CsvReader,
    targets_reader: datasets.CsvReader,
    model_alias: registries.Version,
    metric: metrics.SklearnMetric,
) -> None:
    # given
    run_config = mlflow_service.RunConfig(
        name="EvaluationsTest",
        tags={"context": "evaluations"},
        description="Evaluations job.",
    )
    client = mlflow_service.client()
    # when
    job = jobs.EvaluationsJob(
        logger_service=logger_service,
        mlflow_service=mlflow_service,
        run_config=run_config,
        inputs=inputs_reader,
        targets=targets_reader,
        alias_or_version=model_alias.aliases[0],
        metrics=[metric],
        thresholds=thresholds,
    )
    with job as runner:
        out = runner.run()
    # then
    if should_fail:
        pytest.fail("Should have raised ValueError before reaching this point.")
    expected_keys = {
        "self",
        "logger",
        "client",
        "run",
        "inputs",
        "targets",
        "model_uri",
        "model",
        "eval_data",
        "extra_metrics",
        "result",
        "evaluations_metrics",
    }
    assert expected_keys.issubset(out), (
        "Run should return the expected local variables!"
    )
    # - run
    assert out["run"].info.run_name == run_config.name, "Run name should be the same!"
    # - model uri
    assert model_alias.aliases[0] in out["model_uri"], (
        "Model URI should contain the alias!"
    )
    assert mlflow_service.registry_name in out["model_uri"], (
        "Model URI should contain the registry name!"
    )
    # - evaluations
    assert "R2_test" in out["evaluations_metrics"], (
        "Metric should be logged by mlflow.evaluate!"
    )
    # - mlflow tracking: 1 training-like run (from model_alias fixture) + 1 evaluations run
    experiment = client.get_experiment_by_name(name=mlflow_service.experiment_name)
    assert experiment is not None, "Mlflow experiment should exist!"
    runs = client.search_runs(experiment_ids=experiment.experiment_id)
    assert len(runs) == 2, (
        "There should be two Mlflow runs: the saved model run and evaluations!"
    )
