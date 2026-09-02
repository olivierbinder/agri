# %% IMPORTS

import pytest

from agri import jobs
from agri.core import models
from agri.io import datasets, registries, services

# %% JOBS


@pytest.mark.parametrize("use_alias", [False, True])
def test_explanations_job(
    use_alias: bool,
    mlflow_service: services.MlflowService,
    logger_service: services.LoggerService,
    inputs_samples_reader: datasets.CsvReader,
    tmp_models_explanations_writer: datasets.CsvWriter,
    tmp_samples_explanations_writer: datasets.CsvWriter,
    model_alias: registries.Version,
    loader: registries.CustomLoader,
) -> None:
    # given
    alias_or_version = model_alias.aliases[0] if use_alias else int(model_alias.version)
    # when
    job = jobs.ExplanationsJob(
        logger_service=logger_service,
        mlflow_service=mlflow_service,
        inputs_samples=inputs_samples_reader,
        models_explanations=tmp_models_explanations_writer,
        samples_explanations=tmp_samples_explanations_writer,
        alias_or_version=alias_or_version,
        loader=loader,
    )
    with job as runner:
        out = runner.run()
    # then
    expected_keys = {
        "self",
        "logger",
        "inputs_samples",
        "model_uri",
        "model",
        "models_explanations",
        "samples_explanations",
    }
    assert expected_keys.issubset(out), (
        "Run should return the expected local variables!"
    )
    assert out["inputs_samples"].ndim == 2, "Inputs samples should be a dataframe!"
    assert str(alias_or_version) in out["model_uri"], (
        "Model URI should contain the model alias/version!"
    )
    assert isinstance(out["model"], models.Model), (
        "Unwrapped model should be an instance of a project Model!"
    )
    assert len(out["models_explanations"].index) >= len(
        out["inputs_samples"].columns
    ), "Model explanations should have at least as many rows as input features!"
    assert len(out["samples_explanations"].index) == len(out["inputs_samples"].index), (
        "Samples explanations should have one row per input sample!"
    )
