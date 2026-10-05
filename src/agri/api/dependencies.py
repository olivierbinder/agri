import os

import pandas as pd

from agri.io import registries, services

# Set by the API's Docker image to point at a bundled model directory (see
# `just docker-export-model` and the Dockerfile), bypassing the Mlflow registry
# entirely. Unset for local dev, where the registry (started below) is used instead.
MODEL_URI = os.environ.get("MODEL_URI")

# Real 2013 yields (the model's held-out test year, never trained on) — bundled
# alongside the model so /predict and /recommend can show "actual vs predicted".
ACTUALS_PATH = os.environ.get("ACTUALS_PATH", "deploy/reference/actuals.csv")

# We load the model and the actuals once when the app starts.
_MODEL: registries.Loader.Adapter | None = None
_ACTUALS: pd.DataFrame | None = None


def get_model() -> registries.Loader.Adapter:
    global _MODEL
    if _MODEL is None:
        loader = registries.CustomLoader()
        if MODEL_URI:
            model_uri = MODEL_URI
        else:
            services.MlflowService().start()
            model_uri = registries.uri_for_model_alias_or_version(
                name="agri", alias_or_version="Champion"
            )
        _MODEL = loader.load(uri=model_uri)
    return _MODEL


def get_actuals() -> pd.DataFrame:
    global _ACTUALS
    if _ACTUALS is None:
        _ACTUALS = pd.read_csv(ACTUALS_PATH)
    return _ACTUALS
