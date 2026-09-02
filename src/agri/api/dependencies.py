import os

from agri.io import registries, services

# Set by the API's Docker image to point at a bundled model directory (see
# `just docker-export-model` and the Dockerfile), bypassing the Mlflow registry
# entirely. Unset for local dev, where the registry (started below) is used instead.
MODEL_URI = os.environ.get("MODEL_URI")

# We load the model once when the app starts.
_MODEL: registries.Loader.Adapter | None = None


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
