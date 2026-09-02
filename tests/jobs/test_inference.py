# %% IMPORTS

import pytest

from agri import jobs
from agri.io import datasets, registries, services

# %% JOBS


@pytest.mark.parametrize("use_alias", [False, True])
def test_inference_job(
    use_alias: bool,
    mlflow_service: services.MlflowService,
    logger_service: services.LoggerService,
    inputs_reader: datasets.CsvReader,
    tmp_outputs_writer: datasets.CsvWriter,
    model_alias: registries.Version,
    loader: registries.CustomLoader,
) -> None:
    # given
    alias_or_version = model_alias.aliases[0] if use_alias else int(model_alias.version)
    # when
    job = jobs.InferenceJob(
        logger_service=logger_service,
        mlflow_service=mlflow_service,
        inputs=inputs_reader,
        outputs=tmp_outputs_writer,
        alias_or_version=alias_or_version,
        loader=loader,
    )
    with job as runner:
        out = runner.run()
    # then
    expected_keys = {"self", "logger", "inputs", "model_uri", "model", "outputs"}
    assert expected_keys.issubset(out), (
        "Run should return the expected local variables!"
    )
    assert out["inputs"].ndim == 2, "Inputs should be a dataframe!"
    assert str(alias_or_version) in out["model_uri"], (
        "Model URI should contain the model alias/version!"
    )
    assert mlflow_service.registry_name in out["model_uri"], (
        "Model URI should contain the registry name!"
    )
    assert out["model"].model.metadata.run_id == model_alias.run_id, (
        "Loaded model run id should be the same as the aliased version!"
    )
    assert out["outputs"].ndim == 2, "Outputs should be a dataframe!"
    assert len(out["outputs"]) == len(out["inputs"]), (
        "Outputs should have one row per input!"
    )
