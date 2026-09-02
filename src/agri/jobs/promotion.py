"""Define a job for promoting a registered model version with an alias."""

# %% IMPORTS

import typing as T

from agri.jobs import base

# %% JOBS


class PromotionJob(base.Job):
    """Define a job for promoting a registered model version with an alias.

    https://mlflow.org/docs/latest/model-registry.html#concepts

    Parameters:
        alias (str): the mlflow alias to transition the registered model version.
        version (int | None): the model version to transition (use None for latest).
    """

    KIND: T.Literal["PromotionJob"] = "PromotionJob"

    alias: str = "Champion"
    version: int | None = None
    metric_name: str = "R2"
    greater_is_better: bool = True

    @T.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)
        # - mlflow
        client = self.mlflow_service.client()
        logger.info("With client: {}", client)
        name = self.mlflow_service.registry_name
        # version
        if self.version is None:  # find the best evaluated model version
            logger.info(
                "Search for best evaluated model version using metric: {}_test",
                self.metric_name,
            )
            versions = client.search_model_versions(f"name='{name}'")
            best_version = None
            best_score = float("-inf") if self.greater_is_better else float("inf")

            for v in versions:
                metric_key = f"{self.metric_name}_test"
                if metric_key in v.tags:
                    score = float(v.tags[metric_key])
                    if (
                        self.greater_is_better
                        and score > best_score
                        or not self.greater_is_better
                        and score < best_score
                    ):
                        best_score, best_version = score, v.version

            if best_version is None:
                logger.warning(
                    "No evaluated model found! Using latest version instead."
                )
                version = client.search_model_versions(
                    f"name='{name}'", max_results=1, order_by=["version_number DESC"]
                )[0].version
            else:
                version = best_version
                logger.info(
                    "Found best version {} with {}_test = {}",
                    version,
                    self.metric_name,
                    best_score,
                )
        else:
            version = self.version
        logger.info("From version: {}", version)
        # alias
        logger.info("To alias: {}", self.alias)
        # promote
        logger.info("Promote model: {}", name)
        client.set_registered_model_alias(
            name=name, alias=self.alias, version=str(version)
        )
        model_version = client.get_model_version_by_alias(name=name, alias=self.alias)
        logger.debug("- Model version: {}", model_version)
        # notify
        return locals()
