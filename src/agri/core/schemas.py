"""Define and validate dataframe schemas."""

# %% IMPORTS

import typing as T

import pandas as pd
import pandera as pa
import pandera.typing as papd  # Pandera pandas.
import pandera.typing.common as padt  # Pandera types.

# %% TYPES

# Generic type for a dataframe container
TSchema = T.TypeVar("TSchema", bound="pa.DataFrameModel")

# %% SCHEMAS


class Schema(pa.DataFrameModel):
    """Base class for a dataframe schema.

    Use a schema to type your dataframe object.
    e.g., to communicate and validate its fields.
    """

    class Config:
        """Default configurations for all schemas.

        Parameters:
            coerce (bool): convert data type if possible.
            strict (bool): ensure the data type is correct.
        """

        coerce: bool = True
        strict: bool | str = "filter"

    @classmethod
    def check(cls, data: pd.DataFrame) -> papd.DataFrame[T.Self]:
        """Check the dataframe with this schema.

        Args:
            data (pd.DataFrame): dataframe to check.

        Returns:
            papd.DataFrame[TSchema]: validated dataframe.
        """
        return cls.validate(data)


class InputsSchema(Schema):
    """Schema for the project inputs."""

    Area: papd.Series[str] = pa.Field()
    Item: papd.Series[str] = pa.Field()
    Year: papd.Series[padt.UInt16] = pa.Field(ge=1900)  # ty: ignore
    average_rain_fall_mm_per_year: papd.Series[padt.Float32] = pa.Field(ge=0)  # ty: ignore
    pesticides_tonnes: papd.Series[padt.Float32] = pa.Field(ge=0)  # ty: ignore
    avg_temp: papd.Series[padt.Float32] = pa.Field()  # ty: ignore


Inputs = papd.DataFrame[InputsSchema]


class TargetsSchema(Schema):
    """Schema for the project target."""

    hg_ha_yield: papd.Series[padt.Float32] = pa.Field(alias="hg/ha_yield", ge=0)  # ty: ignore


Targets = papd.DataFrame[TargetsSchema]


class OutputsSchema(Schema):
    """Schema for the project output."""

    prediction: papd.Series[padt.Float32] = pa.Field(ge=0)  # ty: ignore


Outputs = papd.DataFrame[OutputsSchema]


class SHAPValuesSchema(Schema):
    """Schema for the project shap values."""

    class Config:
        """Default configurations this schema.

        Parameters:
            dtype (str): dataframe default data type.
            strict (bool): ensure the data type is correct.
        """

        dtype: str = "float32"
        strict: bool = False


SHAPValues = papd.DataFrame[SHAPValuesSchema]


class FeatureImportancesSchema(Schema):
    """Schema for the project feature importances."""

    feature: papd.Series[str] = pa.Field()
    importance: papd.Series[padt.Float32] = pa.Field()  # ty: ignore


FeatureImportances = papd.DataFrame[FeatureImportancesSchema]
