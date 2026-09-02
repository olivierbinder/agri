"""Configuration for the tests."""

# %% IMPORTS

import os
import typing as T

import pytest
from _pytest import logging as pl

from agri.core import metrics, models, schemas
from agri.io import datasets, registries, services
from agri.utils import searchers, signers, splitters

# %% FIXTURES

# %% - Paths


@pytest.fixture(scope="session")
def tests_path() -> str:
    """Return the path of the tests folder."""
    file = os.path.abspath(__file__)
    return os.path.dirname(file)


@pytest.fixture(scope="session")
def data_path(tests_path: str) -> str:
    """Return the path of the tests data folder."""
    return os.path.join(tests_path, "data")


@pytest.fixture(scope="session")
def inputs_path(data_path: str) -> str:
    """Return the path of the sample inputs dataset."""
    return os.path.join(data_path, "inputs_sample.csv")


@pytest.fixture(scope="session")
def targets_path(data_path: str) -> str:
    """Return the path of the sample targets dataset."""
    return os.path.join(data_path, "targets_sample.csv")


@pytest.fixture(scope="function")
def tmp_outputs_path(tmp_path: str) -> str:
    """Return a tmp path for an outputs dataset."""
    return os.path.join(tmp_path, "outputs.csv")


@pytest.fixture(scope="function")
def tmp_models_explanations_path(tmp_path: str) -> str:
    """Return a tmp path for the model explanations dataset."""
    return os.path.join(tmp_path, "models_explanations.csv")


@pytest.fixture(scope="function")
def tmp_samples_explanations_path(tmp_path: str) -> str:
    """Return a tmp path for the samples explanations dataset."""
    return os.path.join(tmp_path, "samples_explanations.csv")


# %% - Datasets


@pytest.fixture(scope="session")
def inputs_reader(inputs_path: str) -> datasets.CsvReader:
    """Return a reader for the sample inputs dataset."""
    return datasets.CsvReader(path=inputs_path)


@pytest.fixture(scope="session")
def inputs_samples_reader(inputs_path: str) -> datasets.CsvReader:
    """Return a reader for a small subset of the inputs dataset."""
    return datasets.CsvReader(path=inputs_path, limit=50)


@pytest.fixture(scope="session")
def targets_reader(targets_path: str) -> datasets.CsvReader:
    """Return a reader for the sample targets dataset."""
    return datasets.CsvReader(path=targets_path)


@pytest.fixture(scope="function")
def tmp_outputs_writer(tmp_outputs_path: str) -> datasets.CsvWriter:
    """Return a writer for the tmp outputs dataset."""
    return datasets.CsvWriter(path=tmp_outputs_path)


@pytest.fixture(scope="function")
def tmp_models_explanations_writer(
    tmp_models_explanations_path: str,
) -> datasets.CsvWriter:
    """Return a writer for the tmp model explanations dataset."""
    return datasets.CsvWriter(path=tmp_models_explanations_path)


@pytest.fixture(scope="function")
def tmp_samples_explanations_writer(
    tmp_samples_explanations_path: str,
) -> datasets.CsvWriter:
    """Return a writer for the tmp samples explanations dataset."""
    return datasets.CsvWriter(path=tmp_samples_explanations_path)


# %% - Dataframes


@pytest.fixture(scope="session")
def inputs(inputs_reader: datasets.CsvReader) -> schemas.Inputs:
    """Return the sample inputs data."""
    return schemas.InputsSchema.check(inputs_reader.read())


@pytest.fixture(scope="session")
def inputs_samples(inputs_samples_reader: datasets.CsvReader) -> schemas.Inputs:
    """Return a small subset of the sample inputs data."""
    return schemas.InputsSchema.check(inputs_samples_reader.read())


@pytest.fixture(scope="session")
def targets(targets_reader: datasets.CsvReader) -> schemas.Targets:
    """Return the sample targets data."""
    return schemas.TargetsSchema.check(targets_reader.read())


# %% - Splitters


@pytest.fixture(scope="session")
def train_test_splitter() -> splitters.TrainTestSplitter:
    """Return the default train/test splitter."""
    return splitters.TrainTestSplitter(test_size=0.2, random_state=42)


# %% - Searchers


@pytest.fixture(scope="session")
def searcher() -> searchers.GridCVSearcher:
    """Return a small, fast searcher for testing."""
    param_grid = {"max_depth": [2, 3], "n_estimators": [5]}
    return searchers.GridCVSearcher(param_grid=param_grid)


# %% - Subsets


@pytest.fixture(scope="session")
def train_test_sets(
    train_test_splitter: splitters.TrainTestSplitter,
    inputs: schemas.Inputs,
    targets: schemas.Targets,
) -> tuple[schemas.Inputs, schemas.Targets, schemas.Inputs, schemas.Targets]:
    """Return the inputs/targets train and test sets from the splitter."""
    train_index, test_index = next(
        train_test_splitter.split(inputs=inputs, targets=targets)
    )
    inputs_train, inputs_test = inputs.iloc[train_index], inputs.iloc[test_index]
    targets_train, targets_test = targets.iloc[train_index], targets.iloc[test_index]
    return (
        T.cast(schemas.Inputs, inputs_train),
        T.cast(schemas.Targets, targets_train),
        T.cast(schemas.Inputs, inputs_test),
        T.cast(schemas.Targets, targets_test),
    )


# %% - Models


@pytest.fixture(scope="session")
def model(
    train_test_sets: tuple[
        schemas.Inputs, schemas.Targets, schemas.Inputs, schemas.Targets
    ],
) -> models.RandomForest:
    """Return a small, fitted model for testing."""
    inputs_train, targets_train, _, _ = train_test_sets
    model = models.RandomForest(max_depth=3, n_estimators=5, random_state=0)
    model.fit(inputs=inputs_train, targets=targets_train)
    return model


@pytest.fixture(scope="session")
def outputs(model: models.RandomForest, inputs: schemas.Inputs) -> schemas.Outputs:
    """Return the model predictions on the sample inputs."""
    return model.predict(inputs=inputs)


# %% - Metrics


@pytest.fixture(scope="session")
def metric() -> metrics.SklearnMetric:
    """Return the default metric."""
    return metrics.SklearnMetric(
        name="R2", sklearn_name="r2_score", greater_is_better=True
    )


# %% - Signers


@pytest.fixture(scope="session")
def signer() -> signers.InferSigner:
    """Return a model signer."""
    return signers.InferSigner()


@pytest.fixture(scope="session")
def signature(
    signer: signers.InferSigner, inputs: schemas.Inputs, outputs: schemas.Outputs
) -> signers.Signature:
    """Return the signature for the testing model."""
    return signer.sign(inputs=inputs, outputs=outputs)


# %% - Services


@pytest.fixture(scope="session", autouse=True)
def logger_service() -> T.Generator[services.LoggerService, None, None]:
    """Return and start the logger service."""
    service = services.LoggerService(colorize=False, diagnose=True)
    service.start()
    yield service
    service.stop()


@pytest.fixture
def logger_caplog(
    caplog: pl.LogCaptureFixture, logger_service: services.LoggerService
) -> T.Generator[pl.LogCaptureFixture, None, None]:
    """Extend pytest caplog fixture with the logger service (loguru)."""
    # https://loguru.readthedocs.io/en/stable/resources/migration.html#replacing-caplog-fixture-from-pytest-library
    logger = logger_service.logger()
    handler_id = logger.add(
        caplog.handler,
        level=0,
        format="{message}",
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(scope="function", autouse=True)
def mlflow_service(tmp_path: str) -> T.Generator[services.MlflowService, None, None]:
    """Return and start an isolated mlflow service for each test."""
    service = services.MlflowService(
        tracking_uri=f"sqlite:///{tmp_path}/tracking.db",
        registry_uri=f"sqlite:///{tmp_path}/tracking.db",
        experiment_name="Experiment-Testing",
        registry_name="Registry-Testing",
    )
    service.start()
    yield service
    service.stop()


# %% - Registries


@pytest.fixture(scope="session")
def saver() -> registries.CustomSaver:
    """Return the default model saver."""
    return registries.CustomSaver(path="custom-model")


@pytest.fixture(scope="session")
def loader() -> registries.CustomLoader:
    """Return the default model loader."""
    return registries.CustomLoader()


@pytest.fixture(scope="session")
def register() -> registries.MlflowRegister:
    """Return the default model register."""
    return registries.MlflowRegister(tags={"context": "test", "role": "fixture"})


@pytest.fixture(scope="function")
def model_version(
    model: models.RandomForest,
    inputs: schemas.Inputs,
    signature: signers.Signature,
    saver: registries.CustomSaver,
    register: registries.MlflowRegister,
    mlflow_service: services.MlflowService,
) -> registries.Version:
    """Save and register the default model version."""
    run_config = mlflow_service.RunConfig(name="Custom-Run")
    with mlflow_service.run_context(run_config=run_config):
        info = saver.save(model=model, signature=signature, input_example=inputs)
        version = register.register(
            name=mlflow_service.registry_name, model_uri=info.model_uri
        )
    return version


@pytest.fixture(scope="function")
def model_alias(
    model_version: registries.Version,
    mlflow_service: services.MlflowService,
) -> registries.Alias:
    """Promote the default model version with an alias."""
    alias = "Promotion"
    client = mlflow_service.client()
    client.set_registered_model_alias(
        name=mlflow_service.registry_name, alias=alias, version=model_version.version
    )
    return client.get_model_version_by_alias(
        name=mlflow_service.registry_name, alias=alias
    )
