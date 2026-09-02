# %% IMPORTS

import os

import pytest

from agri.core import schemas
from agri.io import datasets

# %% READERS


@pytest.mark.parametrize("limit", [None, 50])
def test_csv_reader(limit: int | None, inputs_path: str) -> None:
    # given
    reader = datasets.CsvReader(path=inputs_path, limit=limit)
    # when
    data = reader.read()
    lineage = reader.lineage(name="inputs", data=data)
    # then
    # - data
    assert data.ndim == 2, "Data should be a dataframe!"
    if limit is not None:
        assert len(data) == limit, "Data should have the limit size!"
    # - lineage
    assert lineage.name == "inputs", "Lineage name should be inputs!"
    assert lineage.source.uri == inputs_path, (  # ty: ignore[unresolved-attribute]
        "Lineage source uri should be the inputs path!"
    )
    assert lineage.profile["num_rows"] == len(data), (  # ty: ignore[not-subscriptable]
        "Lineage profile should contain the data row count!"
    )


def test_parquet_reader_writer(targets: schemas.Targets, tmp_path: str) -> None:
    # given
    path = os.path.join(tmp_path, "targets.parquet")
    writer = datasets.ParquetWriter(path=path)
    reader = datasets.ParquetReader(path=path)
    # when
    writer.write(data=targets)
    data = reader.read()
    # then
    assert os.path.exists(path), "Parquet file should be written!"
    assert len(data) == len(targets), "Read data should have the same row count!"


# %% WRITERS


def test_csv_writer(targets: schemas.Targets, tmp_outputs_path: str) -> None:
    # given
    writer = datasets.CsvWriter(path=tmp_outputs_path)
    # when
    writer.write(data=targets)
    # then
    assert os.path.exists(tmp_outputs_path), "Data should be written!"
