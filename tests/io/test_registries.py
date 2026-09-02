# %% IMPORTS

import mlflow
import pytest

from agri.core import models, schemas
from agri.io import registries, services
from agri.utils import signers

# %% HELPERS


def test_uri_for_model_alias() -> None:
    # given
    name, alias = "testing", "Champion"
    # when
    uri = registries.uri_for_model_alias(name=name, alias=alias)
    # then
    assert uri == f"models:/{name}@{alias}", "The model URI should be valid!"


def test_uri_for_model_version() -> None:
    # given
    name, version = "testing", 1
    # when
    uri = registries.uri_for_model_version(name=name, version=version)
    # then
    assert uri == f"models:/{name}/{version}", "The model URI should be valid!"


def test_uri_for_model_alias_or_version() -> None:
    # given
    name, alias, version = "testing", "Champion", 1
    # when
    alias_uri = registries.uri_for_model_alias_or_version(
        name=name, alias_or_version=alias
    )
    version_uri = registries.uri_for_model_alias_or_version(
        name=name, alias_or_version=version
    )
    # then
    assert alias_uri == registries.uri_for_model_alias(name=name, alias=alias)
    assert version_uri == registries.uri_for_model_version(name=name, version=version)


# %% SAVERS/LOADERS/REGISTERS


def test_custom_pipeline(
    model: models.RandomForest,
    inputs: schemas.Inputs,
    signature: signers.Signature,
    mlflow_service: services.MlflowService,
) -> None:
    # given
    path, name = "custom", "Custom"
    tags = {"registry": "mlflow"}
    saver = registries.CustomSaver(path=path)
    loader = registries.CustomLoader()
    register = registries.MlflowRegister(tags=tags)
    run_config = mlflow_service.RunConfig(name="Custom-Run")
    # when
    with mlflow_service.run_context(run_config=run_config) as run:
        info = saver.save(model=model, signature=signature, input_example=inputs)
        version = register.register(name=name, model_uri=info.model_uri)
    model_uri = registries.uri_for_model_version(name=name, version=version.version)
    adapter = loader.load(uri=model_uri)
    outputs = adapter.predict(inputs=inputs)
    # then
    assert model_uri == f"models:/{name}/{version.version}"
    assert info.run_id == run.info.run_id, "The run id should be the same!"
    assert info.flavors.get("python_function"), "The model should have a pyfunc flavor!"
    assert version.name == name, "The model version name should be the same!"
    assert version.tags == tags, "The model version tags should be the same!"
    assert version.run_id == run.info.run_id, (
        "The model version run id should be the same!"
    )
    assert adapter.model.metadata.run_id == version.run_id, (
        "The adapter model run id should be the same!"
    )
    assert schemas.OutputsSchema.check(outputs) is not None, "Outputs should be valid!"


def test_builtin_pipeline__unsupported_with_custom_transformer(
    model: models.RandomForest,
    inputs: schemas.Inputs,
    signature: signers.Signature,
    mlflow_service: services.MlflowService,
) -> None:
    """BuiltinSaver is not used in production (CustomSaver is the default), and this
    documents why: mlflow.sklearn's default "skops" serialization format refuses to
    (de)serialize our custom AgriFeatureEngineer transformer as an "untrusted type".
    """
    # given
    saver = registries.BuiltinSaver(path="builtin", flavor="sklearn")
    run_config = mlflow_service.RunConfig(name="Builtin-Run")
    # when / then
    with (
        mlflow_service.run_context(run_config=run_config),
        pytest.raises(mlflow.exceptions.MlflowException, match="untrusted types"),
    ):
        saver.save(model=model, signature=signature, input_example=inputs)
