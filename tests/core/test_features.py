# %% IMPORTS

from agri.core import features, schemas

# %% FEATURE ENGINEER


def test_agri_feature_engineer(inputs: schemas.Inputs) -> None:
    # given
    engineer = features.AgriFeatureEngineer()
    # when
    engineer.fit(X=inputs)
    output = engineer.transform(X=inputs)
    feature_names_out = engineer.get_feature_names_out()
    # then
    added = {"rain_temp_interaction", "rain_efficiency", "temp_deviation"}
    assert added.issubset(output.columns), (
        "Transformed data should contain the engineered columns!"
    )
    assert set(feature_names_out) == set(inputs.columns) | added, (
        "Feature names out should be the original columns plus the engineered ones!"
    )
    assert len(output) == len(inputs), "Transform should not change the row count!"
    assert (
        output["rain_temp_interaction"]
        == inputs["average_rain_fall_mm_per_year"] * inputs["avg_temp"]
    ).all(), "Rain-temp interaction should be the product of rainfall and temperature!"


# %% PREPROCESSOR


def test_agri_preprocessor(inputs: schemas.Inputs) -> None:
    # given
    preprocessor = features.AgriPreprocessor()
    transformer = preprocessor.build_transformer(random_state=0)
    # when
    transformed = transformer.fit_transform(inputs, inputs["avg_temp"])
    # then
    n_expected_columns = len(preprocessor.categoricals) + len(preprocessor.numericals)
    assert transformed.shape == (len(inputs), n_expected_columns), (
        "Transformed data should have one column per categorical/numerical feature!"
    )
