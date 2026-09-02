# %% IMPORTS

from agri.core import models, schemas
from agri.io import datasets

# %% SCHEMAS


def test_inputs_schema(inputs_reader: datasets.CsvReader) -> None:
    # given
    schema = schemas.InputsSchema
    # when
    data = inputs_reader.read()
    # then
    assert schema.check(data) is not None, "Inputs data should be valid!"


def test_targets_schema(targets_reader: datasets.CsvReader) -> None:
    # given
    schema = schemas.TargetsSchema
    # when
    data = targets_reader.read()
    # then
    assert schema.check(data) is not None, "Targets data should be valid!"


def test_outputs_schema(outputs: schemas.Outputs) -> None:
    # given
    schema = schemas.OutputsSchema
    # then
    assert schema.check(outputs) is not None, "Outputs data should be valid!"


def test_shap_values_schema(
    model: models.RandomForest, inputs_samples: schemas.Inputs
) -> None:
    # given
    schema = schemas.SHAPValuesSchema
    # when
    data = model.explain_samples(inputs=inputs_samples)
    # then
    assert schema.check(data) is not None, "SHAP values data should be valid!"


def test_feature_importances_schema(model: models.RandomForest) -> None:
    # given
    schema = schemas.FeatureImportancesSchema
    # when
    data = model.explain_model()
    # then
    assert schema.check(data) is not None, "Feature importance data should be valid!"
