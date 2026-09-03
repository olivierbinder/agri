"""Define a job for training and registring a single AI/ML model."""

# %% IMPORTS

import typing as T

import mlflow
import pydantic as pdt

from agri.core import metrics as metrics_
from agri.core import models, schemas
from agri.io import datasets, registries, services
from agri.jobs import base
from agri.utils import signers, splitters

# %% JOBS


class TrainingJob(base.Job):
    """Train and register a single AI/ML model.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
        inputs (datasets.ReaderKind): reader for the inputs data.
        targets (datasets.ReaderKind): reader for the targets data.
        model (models.ModelKind): machine learning model to train. Its params are
            used as-is when use_best_from_tuning is False.
        metrics (metrics_.MetricsKind): metric list to compute.
        use_best_from_tuning (bool): overwrite model params with the ones from the
            most recent Tuning run in this experiment, if any.
        training_window_years (int | None): number of most recent distinct years
            to train on. None trains on 100% of the given data.
        saver (registries.SaverKind): model saver.
        signer (signers.SignerKind): model signer.
        registry (registries.RegisterKind): model register.
    """

    KIND: T.Literal["TrainingJob"] = "TrainingJob"

    # Run
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(
        name="Training"
    )
    # Data
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    targets: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    # Model
    use_best_from_tuning: bool = False
    model: models.ModelKind = pdt.Field(models.RandomForest(), discriminator="KIND")
    # Metrics
    metrics: metrics_.MetricsKind = [metrics_.SklearnMetric()]
    # Training window: TrainingJob still fits on 100% of the data it's given, but
    # this optionally restricts that data to the N most recent distinct years first
    # (cross-validation showed old data hurts accuracy, see RollingWindowSplitter).
    training_window_years: int | None = None
    # Saver
    saver: registries.SaverKind = pdt.Field(
        registries.CustomSaver(), discriminator="KIND"
    )
    # Signer
    signer: signers.SignerKind = pdt.Field(signers.InferSigner(), discriminator="KIND")
    # Registrer
    # - avoid shadowing pydantic `register` pydantic function
    registry: registries.RegisterKind = pdt.Field(
        registries.MlflowRegister(), discriminator="KIND"
    )

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
            client.set_tag(run.info.run_id, "model_kind", self.model.KIND)
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
            # - training window
            client.set_tag(
                run.info.run_id,
                "training_window_years",
                str(self.training_window_years)
                if self.training_window_years is not None
                else "all",
            )
            client.log_param(
                run.info.run_id, "training_window_years", self.training_window_years
            )
            if self.training_window_years is not None:
                years = splitters.distinct_years(inputs)
                window_years = years[-self.training_window_years :]
                mask = inputs["Year"].astype(int).isin(window_years).to_numpy()
                inputs = inputs.loc[mask].reset_index(drop=True)
                targets = targets.loc[mask].reset_index(drop=True)
                logger.info(
                    "Restricted training data to years: {}", sorted(window_years)
                )
                logger.debug("- Inputs shape after window: {}", inputs.shape)
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
            # best from tuning
            logger.info("With model: {}", self.model)
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
            if self.use_best_from_tuning:
                # Latest run, not best-ever RMSE_tune_mean_best: scores from
                # different Tuning runs aren't comparable once the splitter (or its
                # window) changes, so ranking by score can silently resurrect
                # params tuned for a stale regime (see RollingWindowSplitter).
                logger.info("Fetch model params from the most recent Tuning run")
                runs = client.search_runs(
                    experiment_ids=[run.info.experiment_id],
                    filter_string="tags.mlflow.runName = 'Tuning'",
                    order_by=["start_time DESC"],
                    max_results=1,
                )
                if runs:
                    latest_run = runs[0]
                    # Extracted params are logged without a prefix
                    best_params = {}
                    for key, val in latest_run.data.params.items():
                        if hasattr(self.model, key):
                            # Cast to correct type based on current field
                            orig_val = getattr(self.model, key)
                            if isinstance(orig_val, int):
                                best_params[key] = int(val)
                            elif isinstance(orig_val, float):
                                best_params[key] = float(val)
                            else:
                                best_params[key] = val
                    if best_params:
                        logger.info(
                            "Found params from run {}: {}",
                            latest_run.info.run_id,
                            best_params,
                        )
                        self.model.set_params(**best_params)
                else:
                    logger.warning(
                        "No Tuning runs found! Proceeding with configured model params."
                    )

            # log all final model parameters
            for k, v in self.model.get_params().items():
                client.log_param(run_id=run.info.run_id, key=k, value=v)
            # model
            logger.info("Fit model on 100% of the (windowed) data: {}", self.model)
            self.model.fit(inputs=inputs, targets=targets)
            # outputs (on train data, just for sanity check)
            logger.info("Predict outputs on train data: {}", len(inputs))
            outputs = self.model.predict(inputs=inputs)
            logger.debug("- Outputs shape: {}", outputs.shape)
            # metrics (training metrics)
            train_metrics = {}
            for i, metric in enumerate(self.metrics, start=1):
                logger.info("{}. Compute training metric: {}", i, metric)
                score = metric.score(targets=targets, outputs=outputs)
                client.log_metric(
                    run_id=run.info.run_id, key=f"{metric.name}_train", value=score
                )
                train_metrics[metric.name] = score
                logger.debug("- Metric score: {}", score)
            # signer
            logger.info("Sign model: {}", self.signer)
            model_signature = self.signer.sign(inputs=inputs, outputs=outputs)
            logger.debug("- Model signature: {}", model_signature.to_dict())
            # saver
            logger.info("Save model: {}", self.saver)
            model_info = self.saver.save(
                model=self.model, signature=model_signature, input_example=inputs
            )
            logger.debug("- Model URI: {}", model_info.model_uri)
            # register
            logger.info("Register model: {}", self.registry)
            model_version = self.registry.register(
                name=self.mlflow_service.registry_name, model_uri=model_info.model_uri
            )
            logger.debug("- Model version: {}", model_version)

            # tag model version with training metrics
            logger.info("Tag model version with training metrics")
            for metric_name, score in train_metrics.items():
                client.set_model_version_tag(
                    name=self.mlflow_service.registry_name,
                    version=model_version.version,
                    key=f"{metric_name}_train",
                    value=str(score),
                )

            # notify
        return locals()
