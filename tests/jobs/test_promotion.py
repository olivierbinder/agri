# %% IMPORTS

import pytest

from agri import jobs
from agri.io import registries, services

# %% JOBS


@pytest.mark.parametrize("explicit_version", [False, True])
def test_promotion_job(
    explicit_version: bool,
    mlflow_service: services.MlflowService,
    logger_service: services.LoggerService,
    model_version: registries.Version,
) -> None:
    # given
    alias = "Testing"
    version = int(model_version.version) if explicit_version else None
    # when
    job = jobs.PromotionJob(
        logger_service=logger_service,
        mlflow_service=mlflow_service,
        version=version,
        alias=alias,
    )
    with job as runner:
        out = runner.run()
    # then
    expected_keys = {"self", "logger", "client", "name", "version", "model_version"}
    assert expected_keys.issubset(out), (
        "Run should return the expected local variables!"
    )
    assert out["name"] == mlflow_service.registry_name, "Model name should be the same!"
    assert str(out["version"]) == str(model_version.version), (
        "Resolved version should point to the only registered model version!"
    )
    assert out["model_version"].aliases == [alias], (
        "Model version aliases should contain the given alias!"
    )
    assert out["model_version"].run_id == model_version.run_id, (
        "Model version run id should be the same!"
    )
