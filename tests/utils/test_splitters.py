# %% IMPORTS

from agri.core import schemas
from agri.utils import splitters

# %% SPLITTERS


def test_train_test_splitter(inputs: schemas.Inputs, targets: schemas.Targets) -> None:
    # given
    test_size = 40
    splitter = splitters.TrainTestSplitter(
        shuffle=False, test_size=test_size, random_state=0
    )
    # when
    n_splits = splitter.get_n_splits(inputs=inputs, targets=targets)
    splits = list(splitter.split(inputs=inputs, targets=targets))
    train_index, test_index = splits[0]
    # then
    assert n_splits == len(splits) == 1, "Splitter should return 1 split!"
    assert len(test_index) == test_size, "Test index should have the given size!"
    assert len(train_index) == len(inputs) - test_size, (
        "Train index should have the remaining size!"
    )
    assert not inputs.iloc[test_index].empty, (
        "Test index should be a subset of the inputs!"
    )
    assert not targets.iloc[train_index].empty, (
        "Train index should be a subset of the targets!"
    )


def test_time_series_splitter(inputs: schemas.Inputs, targets: schemas.Targets) -> None:
    # given
    n_splits, test_size = 3, 20
    splitter = splitters.TimeSeriesSplitter(
        gap=0, n_splits=n_splits, test_size=test_size
    )
    # when
    n_splits_reported = splitter.get_n_splits(inputs=inputs, targets=targets)
    splits = list(splitter.split(inputs=inputs, targets=targets))
    # then
    assert n_splits_reported == len(splits) == n_splits, (
        "Splitter should return the given n splits!"
    )
    for train_index, test_index in splits:
        assert len(test_index) == test_size, (
            "Test index should have the given test size!"
        )
        assert train_index.max() < test_index.min(), (
            "Train index should always be lower than test index!"
        )
        assert not inputs.iloc[train_index].empty, (
            "Train index should be a subset of the inputs!"
        )
        assert not inputs.iloc[test_index].empty, (
            "Test index should be a subset of the inputs!"
        )


def test_year_splitter(inputs: schemas.Inputs, targets: schemas.Targets) -> None:
    # given
    splitter = splitters.YearSplitter()
    folds = [
        (1993, 1994, 1997),
        (1997, 1998, 2001),
        (2001, 2002, 2005),
        (2005, 2006, 2009),
    ]
    # when
    n_splits = splitter.get_n_splits(inputs=inputs, targets=targets)
    splits = list(splitter.split(inputs=inputs, targets=targets))
    # then
    assert n_splits == len(splits) == 4, "Splitter should return 4 fixed folds!"
    years = inputs["Year"].astype(int)
    for (train_index, test_index), (train_end, test_start, test_end) in zip(
        splits, folds
    ):
        assert years.iloc[train_index].max() <= train_end, (
            "Train fold should only contain years up to the fold's train_end!"
        )
        assert years.iloc[test_index].between(test_start, test_end).all(), (
            "Test fold should only contain years within the fold's test range!"
        )
        assert set(train_index).isdisjoint(test_index), (
            "Train and test indexes should never overlap!"
        )
