# %% IMPORTS

from agri.core import metrics, models, schemas
from agri.utils import searchers, splitters

# %% SEARCHERS


def test_grid_cv_searcher(
    model: models.RandomForest,
    metric: metrics.Metric,
    inputs: schemas.Inputs,
    targets: schemas.Targets,
    train_test_splitter: splitters.Splitter,
) -> None:
    # given
    param_grid = {"max_depth": [2, 3], "n_estimators": [5]}
    searcher = searchers.GridCVSearcher(param_grid=param_grid, verbose=0)
    # when
    results, best_score, best_params = searcher.search(
        model=model,
        metrics=[metric],
        inputs=inputs,
        targets=targets,
        cv=train_test_splitter,
    )
    # then
    assert set(best_params) == set(param_grid), (
        "Best params should have the same keys as grid!"
    )
    assert float("-inf") < best_score < float("+inf"), (
        "Best score should be a floating number!"
    )
    assert len(results) == 2, "Results should have one row per candidate (2x1 grid)!"
