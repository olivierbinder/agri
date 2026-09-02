"""Split dataframes into subsets (e.g., train/valid/test)."""

# %% IMPORTS

import abc
import typing as T

import numpy as np
import numpy.typing as npt
import pydantic as pdt
from sklearn import model_selection

from agri.core import schemas

# %% TYPES

Index = npt.NDArray[np.int64]
TrainTestIndex = tuple[Index, Index]
TrainTestSplits = T.Iterator[TrainTestIndex]

# %% SPLITTERS


class Splitter(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a splitter.

    Use splitters to split data in sets.
    e.g., split between a train/test subsets.

    # https://scikit-learn.org/stable/glossary.html#term-CV-splitter
    """

    KIND: str

    @abc.abstractmethod
    def split(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> TrainTestSplits:
        """Split a dataframe into subsets.

        Args:
            inputs (schemas.Inputs): model inputs.
            targets (schemas.Targets): model targets.
            groups (Index | None, optional): group labels.

        Returns:
            TrainTestSplits: iterator over the dataframe train/test splits.
        """

    @abc.abstractmethod
    def get_n_splits(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> int:
        """Get the number of splits generated.

        Args:
            inputs (schemas.Inputs): models inputs.
            targets (schemas.Targets): model targets.
            groups (Index | None, optional): group labels.

        Returns:
            int: number of splits generated.
        """


class TrainTestSplitter(Splitter):
    """Split a dataframe into a train and test set.

    Parameters:
        shuffle (bool): shuffle the dataset. Default is False.
        test_size (int | float): number/ratio for the test set.
        random_state (int): random state for the splitter object.
    """

    KIND: T.Literal["TrainTestSplitter"] = "TrainTestSplitter"

    shuffle: bool = True  # shuffle the data
    test_size: int | float = 0.2  # 20% for validation
    random_state: int = 42

    @T.override
    def split(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> TrainTestSplits:
        index = np.arange(len(inputs))  # return integer position
        train_index, test_index = model_selection.train_test_split(
            index,
            shuffle=self.shuffle,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        yield train_index, test_index

    @T.override
    def get_n_splits(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> int:
        return 1


class TimeSeriesSplitter(Splitter):
    """Split a dataframe into fixed time series subsets.

    Parameters:
        gap (int): gap between splits.
        n_splits (int): number of split to generate.
        test_size (int | float): number or ratio for the test dataset.
    """

    KIND: T.Literal["TimeSeriesSplitter"] = "TimeSeriesSplitter"

    gap: int = 0
    n_splits: int = 4
    test_size: int | float = 24 * 30 * 2  # 2 months

    @T.override
    def split(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> TrainTestSplits:
        splitter = model_selection.TimeSeriesSplit(
            n_splits=self.n_splits, test_size=self.test_size, gap=self.gap
        )
        yield from splitter.split(inputs)

    @T.override
    def get_n_splits(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> int:
        return self.n_splits


class YearSplitter(Splitter):
    """Split a dataframe into custom fixed year ranges.

    Fold 1: Train 1990-1993, Test 1994-1997
    Fold 2: Train 1990-1997, Test 1998-2001
    Fold 3: Train 1990-2001, Test 2002-2005
    Fold 4: Train 1990-2005, Test 2006-2009
    """

    KIND: T.Literal["YearSplitter"] = "YearSplitter"

    @T.override
    def split(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> TrainTestSplits:
        folds = [
            # (train_end_year, test_start_year, test_end_year)
            (1993, 1994, 1997),
            (1997, 1998, 2001),
            (2001, 2002, 2005),
            (2005, 2006, 2009),
        ]

        years = inputs["Year"].astype(int)

        for train_end, test_start, test_end in folds:
            train_mask = (years >= 1990) & (years <= train_end)
            test_mask = (years >= test_start) & (years <= test_end)

            # Using np.where to get positional integer indices (required by sklearn)
            train_index = np.where(train_mask)[0]
            test_index = np.where(test_mask)[0]

            yield train_index, test_index

    @T.override
    def get_n_splits(
        self,
        inputs: schemas.Inputs,
        targets: schemas.Targets,
        groups: Index | None = None,
    ) -> int:
        return 4


SplitterKind = TrainTestSplitter | TimeSeriesSplitter | YearSplitter
