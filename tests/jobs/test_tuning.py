# %% IMPORTS

from agri import jobs
from agri.core import metrics, models
from agri.io import datasets, services
from agri.utils import searchers, splitters

# %% JOBS


def test_tuning_job(
    mlflow_service: services.MlflowService,
    logger_service: services.LoggerService,
    inputs_reader: datasets.CsvReader,
    targets_reader: datasets.CsvReader,
    metric: metrics.SklearnMetric,
    train_test_splitter: splitters.TrainTestSplitter,
    searcher: searchers.GridCVSearcher,
) -> None:
    # given
    run_config = mlflow_service.RunConfig(
        name="TuningTest", tags={"context": "tuning"}, description="Tuning job."
    )
    model = models.RandomForest(random_state=0)
    client = mlflow_service.client()
    # when
    job = jobs.TuningJob(
        logger_service=logger_service,
        mlflow_service=mlflow_service,
        run_config=run_config,
        inputs=inputs_reader,
        targets=targets_reader,
        model=model,
        metrics=[metric],
        splitter=train_test_splitter,
        searcher=searcher,
    )
    with job as runner:
        out = runner.run()
    # then
    expected_keys = {
        "self",
        "logger",
        "run",
        "client",
        "inputs",
        "targets",
        "results",
        "best_score",
        "best_params",
    }
    assert expected_keys.issubset(out), (
        "Run should return the expected local variables!"
    )
    # - run
    assert out["run"].info.run_name == run_config.name, "Run name should be the same!"
    # - data
    assert out["inputs"].ndim == 2, "Inputs should be a dataframe!"
    assert out["targets"].ndim == 2, "Targets should be a dataframe!"
    # - results
    assert out["results"].ndim == 2, "Results should be a dataframe!"
    assert len(out["results"]) == 2, "Results should have one row per grid candidate!"
    # - best score / params
    assert float("-inf") < out["best_score"] < float("+inf"), (
        "Best score should be a finite number!"
    )
    assert set(out["best_params"]) == set(searcher.param_grid), (
        "Best params should have the same keys as the search grid!"
    )
    # - mlflow tracking: 1 parent run + 1 child run per grid candidate
    experiment = client.get_experiment_by_name(name=mlflow_service.experiment_name)
    assert experiment is not None, "Mlflow experiment should exist!"
    runs = client.search_runs(experiment_ids=experiment.experiment_id)
    assert len(runs) == len(out["results"]) + 1, (
        "Mlflow should have 1 run per grid candidate, plus the parent run!"
    )
