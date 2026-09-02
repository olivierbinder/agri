# %% IMPORTS

from agri import jobs
from agri.core import metrics, models, schemas
from agri.io import datasets, registries, services
from agri.utils import signers

# %% JOBS


def test_training_job(
    mlflow_service: services.MlflowService,
    logger_service: services.LoggerService,
    inputs_reader: datasets.CsvReader,
    targets_reader: datasets.CsvReader,
    metric: metrics.SklearnMetric,
    saver: registries.CustomSaver,
    signer: signers.InferSigner,
    register: registries.MlflowRegister,
) -> None:
    # given
    run_config = mlflow_service.RunConfig(
        name="TrainingTest", tags={"context": "training"}, description="Training job."
    )
    model = models.RandomForest(max_depth=3, n_estimators=5, random_state=0)
    client = mlflow_service.client()
    # when
    job = jobs.TrainingJob(
        logger_service=logger_service,
        mlflow_service=mlflow_service,
        run_config=run_config,
        inputs=inputs_reader,
        targets=targets_reader,
        model=model,
        metrics=[metric],
        saver=saver,
        signer=signer,
        registry=register,
    )
    with job as runner:
        out = runner.run()
    # then
    expected_keys = {
        "self",
        "logger",
        "client",
        "run",
        "inputs",
        "targets",
        "outputs",
        "model_signature",
        "model_info",
        "model_version",
        "train_metrics",
    }
    assert expected_keys.issubset(out), (
        "Run should return the expected local variables!"
    )
    # - run
    assert run_config.tags is not None, "Run config tags should be set!"
    assert out["run"].info.run_name == run_config.name, "Run name should be the same!"
    assert run_config.description in out["run"].data.tags.values(), (
        "Run description should be in run tags!"
    )
    # - data
    assert out["inputs"].ndim == 2, "Inputs should be a dataframe!"
    assert out["targets"].ndim == 2, "Targets should be a dataframe!"
    assert len(out["inputs"]) == len(out["targets"]), (
        "Inputs and targets should have the same number of rows!"
    )
    # - outputs (fit on 100% of the data, predicted for the sanity check)
    assert schemas.OutputsSchema.check(out["outputs"]) is not None, (
        "Outputs should be valid!"
    )
    assert len(out["outputs"]) == len(out["inputs"]), (
        "Outputs should have one row per input!"
    )
    # - metrics
    assert out["train_metrics"].keys() == {metric.name}, (
        "Train metrics should be keyed by metric name!"
    )
    # - model signature
    assert out["model_signature"].inputs is not None, (
        "Model signature inputs should be set!"
    )
    assert out["model_signature"].outputs is not None, (
        "Model signature outputs should be set!"
    )
    # - model info
    assert out["model_info"].run_id == out["run"].info.run_id, (
        "Model info run id should be the same as the run!"
    )
    # - model version
    assert str(out["model_version"].version) == "1", "Model version number should be 1!"
    assert out["model_version"].tags == register.tags, (
        "Model version tags should be the same!"
    )
    assert out["model_version"].name == mlflow_service.registry_name, (
        "Model name should be the registry name!"
    )
    # - mlflow tracking
    experiment = client.get_experiment_by_name(name=mlflow_service.experiment_name)
    assert experiment is not None, "Mlflow experiment should exist!"
    runs = client.search_runs(experiment_ids=experiment.experiment_id)
    assert len(runs) == 1, "There should be a single Mlflow run for training!"
    assert f"{metric.name}_train" in runs[0].data.metrics, (
        "Training metric should be logged in Mlflow!"
    )
    assert runs[0].info.status == "FINISHED", "Mlflow run status should be FINISHED!"
    # - mlflow registry
    model_version = client.get_model_version(
        name=mlflow_service.registry_name, version=out["model_version"].version
    )
    assert model_version.run_id == out["run"].info.run_id, (
        "Mlflow model version run id should be the same as the run!"
    )
