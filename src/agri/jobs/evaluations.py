"""Define a job for evaluating registered models with data."""

# %% IMPORTS

import typing as T

import mlflow
import pandas as pd
import pydantic as pdt

from agri.core import metrics as metrics_
from agri.core import schemas
from agri.io import datasets, registries, services
from agri.jobs import base

# %% JOBS


class EvaluationsJob(base.Job):
    """Generate evaluations from a registered model and a dataset.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
        inputs (datasets.ReaderKind): reader for the inputs data.
        targets (datasets.ReaderKind): reader for the targets data.
        model_type (str): model type (e.g. "regressor", "classifier").
        alias_or_version (str | int): alias or version for the  model.
        metrics (metrics_.MetricsKind): metric list to compute.
        evaluators (list[str]): list of evaluators to use.
        thresholds (dict[str, metrics_.Threshold] | None): metric thresholds.
    """

    KIND: T.Literal["EvaluationsJob"] = "EvaluationsJob"

    # Run
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(
        name="Evaluations"
    )
    # Data
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    targets: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    # Model
    model_type: str = "regressor"
    alias_or_version: str | int | None = None
    # Loader
    loader: registries.LoaderKind = pdt.Field(
        registries.CustomLoader(), discriminator="KIND"
    )
    # Metrics
    metrics: metrics_.MetricsKind = [metrics_.SklearnMetric()]
    # Evaluators
    evaluators: list[str] = ["default"]
    # Thresholds
    thresholds: dict[str, metrics_.Threshold] = {
        "r2_score": metrics_.Threshold(threshold=0.5, greater_is_better=True)
    }

    @T.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        # - mlflow
        client = self.mlflow_service.client()
        logger.info("With client: {}", client.tracking_uri)
        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            logger.info("With run context: {}", run.info)
            # data
            # - inputs
            logger.info("Read inputs: {}", self.inputs)
            inputs_ = self.inputs.read()  # unchecked!
            inputs = schemas.InputsSchema.check(inputs_)
            logger.debug("- Inputs shape: {}", inputs.shape)
            # - targets
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
            name = self.mlflow_service.registry_name
            logger.info("With model: {}", name)
            if self.alias_or_version is None:
                logger.info("Fetch latest model version")
                version = client.search_model_versions(
                    f"name='{name}'", max_results=1, order_by=["version_number DESC"]
                )[0].version
                alias_or_version = version
            else:
                alias_or_version = self.alias_or_version

            # Fetch model_kind from the original Training run that registered this version
            if str(alias_or_version).isdigit():
                resolved_version = str(alias_or_version)
            else:
                resolved_version = client.get_model_version_by_alias(
                    name=name, alias=str(alias_or_version)
                ).version
            model_version_details = client.get_model_version(
                name=name, version=str(resolved_version)
            )
            original_run = client.get_run(str(model_version_details.run_id))

            # Log tags
            model_kind = original_run.data.tags.get("model_kind", "Unknown")
            client.set_tag(run.info.run_id, "model_kind", model_kind)
            preprocessor_kind = original_run.data.tags.get(
                "preprocessor_kind", "Unknown"
            )
            client.set_tag(run.info.run_id, "preprocessor_kind", preprocessor_kind)

            # Log params from training run to keep track of model parameters
            for key, val in original_run.data.params.items():
                client.log_param(run.info.run_id, key, val)

            model_uri = registries.uri_for_model_alias_or_version(
                name=name,
                alias_or_version=alias_or_version,
            )
            logger.debug("- Model URI: {}", model_uri)
            # loader is no longer strictly needed for predict, but we keep it for reference or if needed
            logger.info("Load model: {}", self.loader)
            model = self.loader.load(uri=model_uri)
            logger.debug("- Model: {}", model)

            # combine data for mlflow.evaluate
            logger.info("Combine inputs and targets for mlflow.evaluate")
            eval_data = pd.concat([inputs, targets], axis=1)

            # prepare mlflow evaluate arguments
            extra_metrics = [m.to_mlflow(suffix="_test") for m in self.metrics]

            # evaluations
            logger.info("Run mlflow.models.evaluate")
            result = mlflow.models.evaluate(
                model=model_uri,
                data=eval_data,
                targets="hg/ha_yield",
                model_type=None,  # None forces MLflow to only use our extra_metrics
                extra_metrics=extra_metrics,
                evaluators=self.evaluators,
            )

            evaluations_metrics = result.metrics
            logger.debug("- Evaluated metrics: {}", evaluations_metrics)

            # validate thresholds manually
            if self.thresholds:
                logger.info("Validating thresholds...")
                for metric_name, threshold_cfg in self.thresholds.items():
                    # metric key could end with _eval if extra_metrics logs it that way
                    # but typically extra_metrics logs with the name we gave it.
                    score = evaluations_metrics.get(metric_name)
                    if score is not None:
                        if (
                            threshold_cfg.greater_is_better
                            and score < threshold_cfg.threshold
                        ):
                            raise ValueError(
                                f"Validation failed: {metric_name} ({score}) < {threshold_cfg.threshold}"
                            )
                        elif (
                            not threshold_cfg.greater_is_better
                            and score > threshold_cfg.threshold
                        ):
                            raise ValueError(
                                f"Validation failed: {metric_name} ({score}) > {threshold_cfg.threshold}"
                            )
                    else:
                        logger.warning(
                            "Threshold metric {} not found in evaluated metrics.",
                            metric_name,
                        )

            # tag model version
            if (
                isinstance(alias_or_version, (int, str))
                and str(alias_or_version).isdigit()
            ):
                logger.info(
                    "Tag model version {} with evaluation metrics", alias_or_version
                )
                for metric_name, metric_value in evaluations_metrics.items():
                    # mlflow tag values must be strings
                    client.set_model_version_tag(
                        name=name,
                        version=str(alias_or_version),
                        key=metric_name,
                        value=str(metric_value),
                    )
            # notify
        return locals()
