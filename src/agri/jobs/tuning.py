"""Define a job for finding the best hyperparameters for a model."""

# %% IMPORTS

import typing as T

import mlflow
import pydantic as pdt

from agri.core import metrics as metrics_
from agri.core import models, schemas
from agri.io import datasets, services
from agri.jobs import base
from agri.utils import searchers, splitters

# %% JOBS


class TuningJob(base.Job):
    """Find the best hyperparameters for a model.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
        inputs (datasets.ReaderKind): reader for the inputs data.
        targets (datasets.ReaderKind): reader for the targets data.
        model (models.ModelKind): machine learning model to tune.
        metrics (metrics_.MetricsKind): metrics to compute.
        splitter (splitters.SplitterKind): data sets splitter.
        searcher: (searchers.SearcherKind): hparams searcher.
    """

    KIND: T.Literal["TuningJob"] = "TuningJob"

    # Run
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(
        name="Tuning"
    )
    # Data
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    targets: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    # Model
    model: models.ModelKind = pdt.Field(models.RandomForest(), discriminator="KIND")
    # Metrics
    metrics: metrics_.MetricsKind = [metrics_.SklearnMetric()]
    # splitter
    splitter: splitters.SplitterKind = pdt.Field(
        splitters.TrainTestSplitter(), discriminator="KIND"
    )
    # Searcher
    searcher: searchers.SearcherKind = pdt.Field(
        searchers.GridCVSearcher(
            param_grid={
                "max_depth": [5, 10, 20],
                "n_estimators": [50, 100],
            }
        ),
        discriminator="KIND",
    )

    @T.override
    def run(self) -> base.Locals:
        """Run the tuning job in context."""
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            logger.info("With run context: {}", run.info)
            client = self.mlflow_service.client()
            client.set_tag(run.info.run_id, "model_kind", self.model.KIND)
            # data : inputs
            logger.info("Read inputs: {}", self.inputs)
            inputs_ = self.inputs.read()  # unchecked!
            inputs = schemas.InputsSchema.check(inputs_)
            logger.debug("- Inputs shape: {}", inputs.shape)
            # data : targets
            logger.info("Read targets: {}", self.targets)
            targets_ = self.targets.read()  # unchecked!
            targets = schemas.TargetsSchema.check(targets_)
            logger.debug("- Targets shape: {}", targets.shape)
            # lineage
            # - inputs
            logger.info("Log lineage: inputs")
            inputs_lineage = self.inputs.lineage(data=inputs, name="inputs")
            mlflow.log_input(dataset=inputs_lineage, context=self.run_config.name)
            logger.debug("- Inputs lineage: {}", inputs_lineage.to_dict())
            # - targets
            logger.info("Log lineage: targets")
            targets_lineage = self.targets.lineage(
                data=targets, name="targets", targets="hg/ha_yield"
            )
            mlflow.log_input(dataset=targets_lineage, context=self.run_config.name)
            logger.debug("- Targets lineage: {}", targets_lineage.to_dict())
            # model
            logger.info("With model: {}", self.model)
            client.set_tag(run.info.run_id, "model_kind", self.model.KIND)
            client.log_param(run.info.run_id, "model_kind", self.model.KIND)
            client.set_tag(
                run.info.run_id, "preprocessor_kind", self.model.preprocessor.KIND
            )
            client.log_param(
                run.info.run_id, "preprocessor_kind", self.model.preprocessor.KIND
            )
            for k, v in self.model.preprocessor.model_dump().items():
                if k != "KIND" and isinstance(v, (int, float, str, bool)):
                    client.log_param(
                        run_id=run.info.run_id, key=f"preprocess_{k}", value=v
                    )
            # metrics
            logger.info("With metrics: {}", self.metrics)
            # splitter
            logger.info("With splitter: {}", self.splitter)
            client.set_tag(run.info.run_id, "splitter_kind", self.splitter.KIND)
            client.log_param(run.info.run_id, "splitter_kind", self.splitter.KIND)
            for k, v in self.splitter.model_dump().items():
                if k != "KIND" and isinstance(v, (int, float, str, bool)):
                    client.log_param(
                        run_id=run.info.run_id, key=f"splitter_{k}", value=v
                    )

            # searcher
            logger.info("Run searcher: {}", self.searcher)
            results, best_score, best_params = self.searcher.search(
                model=self.model,
                metrics=self.metrics,
                inputs=inputs,
                targets=targets,
                cv=self.splitter,
            )
            logger.debug("- Results: {}", results.shape)
            logger.debug("- Best Score: {}", best_score)
            logger.debug("- Best Params: {}", best_params)
            # explicitly log them to mlflow so training can find them
            primary_metric = self.metrics[0]
            primary_metric_name = primary_metric.name
            primary_sign = 1 if primary_metric.greater_is_better else -1
            client.log_metric(
                run_id=run.info.run_id,
                key=f"{primary_metric_name}_tune_mean_best",
                value=best_score * primary_sign,
            )
            for k, v in best_params.items():
                client.log_param(run_id=run.info.run_id, key=k, value=v)

            # create nested runs for each parameter combination
            logger.info("Log child runs for each combination")
            for index, row in results.iterrows():
                with mlflow.start_run(
                    run_name=f"combination_{index}",
                    nested=True,
                    log_system_metrics=False,
                ):
                    # tags (so child runs are filterable on their own, without
                    # needing to look up the parent Tuning run)
                    mlflow.set_tag("model_kind", self.model.KIND)
                    mlflow.set_tag("splitter_kind", self.splitter.KIND)
                    mlflow.set_tag("preprocessor_kind", self.model.preprocessor.KIND)
                    # log parameters
                    mlflow.log_params(row["params"])
                    # log metrics
                    for metric in self.metrics:
                        sign = 1 if metric.greater_is_better else -1
                        mlflow.log_metric(
                            f"{metric.name}_tune_mean",
                            row[f"mean_test_{metric.name}"] * sign,
                        )
                        mlflow.log_metric(
                            f"{metric.name}_tune_std", row[f"std_test_{metric.name}"]
                        )
                    # log fit time
                    mlflow.log_metric("mean_fit_time", row["mean_fit_time"])
            # notify
        return locals()
