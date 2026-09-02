"""Read/Write datasets from/to external sources/destinations."""

# %% IMPORTS

import abc
import pathlib
import typing as T

import mlflow.data.pandas_dataset as lineage
import pandas as pd
import pydantic as pdt

# %% ▏TYPINGS
type Lineage = lineage.PandasDataset

# %% READERS


class Reader(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a dataset reader.

    Use a reader to load a dataset in memory.
    e.g., to read file, database, cloud storage, ...

    Parameters:
        limit (int, optional): maximum number of rows to read. Defaults to None.
    """

    KIND: str

    limit: int | None = None

    @abc.abstractmethod
    def read(self) -> pd.DataFrame:
        """Read a dataframe from a dataset.

        Returns:
            pd.DataFrame: dataframe representation.
        """

    @abc.abstractmethod
    def lineage(
        self,
        name: str,
        data: pd.DataFrame,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        """Generate lineage information.

        Args:
            name (str): dataset name.
            data (pd.DataFrame): reader dataframe.
            targets (str | None): name of the target column.
            predictions (str | None): name of the prediction column.

        Returns:
            Lineage: lineage information.
        """


class ParquetReader(Reader):
    """Read a dataframe from a parquet file.

    Parameters:
        path (str): local path to the dataset.
    """

    KIND: T.Literal["ParquetReader"] = "ParquetReader"

    path: str
    backend: T.Literal["pyarrow", "numpy_nullable"] = "pyarrow"

    @T.override
    def read(self) -> pd.DataFrame:
        # can't limit rows at read time
        data = pd.read_parquet(self.path, dtype_backend=self.backend)
        if self.limit is not None:
            data = data.head(self.limit)
        return data

    @T.override
    def lineage(
        self,
        name: str,
        data: pd.DataFrame,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        return lineage.from_pandas(
            df=data,
            name=name,
            source=self.path,
            targets=targets,
            predictions=predictions,
        )


class CsvReader(Reader):
    """Read a dataframe from a csv file.

    Parameters:
        path (str): local path to the dataset.
    """

    KIND: T.Literal["CsvReader"] = "CsvReader"

    path: str
    index_col: int | None = None

    @T.override
    def read(self) -> pd.DataFrame:
        data = pd.read_csv(self.path, nrows=self.limit, index_col=self.index_col)
        return data

    @T.override
    def lineage(
        self,
        name: str,
        data: pd.DataFrame,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        return lineage.from_pandas(
            df=data,
            name=name,
            source=self.path,
            targets=targets,
            predictions=predictions,
        )


ReaderKind = ParquetReader | CsvReader

# %% WRITERS


class Writer(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a dataset writer.

    Use a writer to save a dataset from memory.
    e.g., to write file, database, cloud storage, ...
    """

    KIND: str

    @abc.abstractmethod
    def write(self, data: pd.DataFrame) -> None:
        """Write a dataframe to a dataset.

        Args:
            data (pd.DataFrame): dataframe representation.
        """


class ParquetWriter(Writer):
    """Writer a dataframe to a parquet file.

    Parameters:
        path (str): local or S3 path to the dataset.
    """

    KIND: T.Literal["ParquetWriter"] = "ParquetWriter"

    path: str

    @T.override
    def write(self, data: pd.DataFrame) -> None:
        pd.DataFrame.to_parquet(data, self.path)


class CsvWriter(Writer):
    """Write data to a CSV file.

    Parameters:
        path (str): path for the file.
    """

    KIND: T.Literal["CsvWriter"] = "CsvWriter"

    path: str

    @T.override
    def write(self, data: pd.DataFrame) -> None:
        path = pathlib.Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)


WriterKind = ParquetWriter | CsvWriter
