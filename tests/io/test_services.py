# %% IMPORTS

import mlflow
from _pytest import logging as pl

from agri.io import services

# %% SERVICES


def test_logger_service(
    logger_service: services.LoggerService, logger_caplog: pl.LogCaptureFixture
) -> None:
    # given
    logger = logger_service.logger()
    # when
    logger.debug("DEBUG")
    logger.error("ERROR")
    # then
    assert "DEBUG" in logger_caplog.messages, "Debug message should be logged!"
    assert "ERROR" in logger_caplog.messages, "Error message should be logged!"


def test_mlflow_service(mlflow_service: services.MlflowService) -> None:
    # given
    run_config = mlflow_service.RunConfig(
        name="testing",
        tags={"service": "mlflow"},
        description="a test run.",
        log_system_metrics=False,
    )
    # when
    client = mlflow_service.client()
    with mlflow_service.run_context(run_config=run_config) as context:
        pass
    finished = client.get_run(run_id=context.info.run_id)
    # then
    # - mlflow
    assert mlflow_service.tracking_uri == mlflow.get_tracking_uri(), (
        "Tracking URI should be the same!"
    )
    assert mlflow.get_experiment_by_name(mlflow_service.experiment_name), (
        "Experiment should be setup!"
    )
    # - client
    assert mlflow_service.tracking_uri == client.tracking_uri, (
        "Tracking URI should be the same!"
    )
    assert client.get_experiment_by_name(mlflow_service.experiment_name), (
        "Experiment should be setup!"
    )
    # - context
    assert context.info.run_name == run_config.name, "Context name should be the same!"
    assert run_config.description in context.data.tags.values(), (
        "Context desc. should be in tags values!"
    )
    assert (
        run_config.tags is not None
        and context.data.tags.items() > run_config.tags.items()
    ), "Context tags should be a subset of the given tags!"
    assert context.info.status == "RUNNING", "Context should be running!"
    # - finished
    assert finished.info.status == "FINISHED", "Finished run should be finished!"
