from agri.io import registries, services

# We load the model once when the app starts.
_MODEL: registries.Loader.Adapter | None = None


def get_model() -> registries.Loader.Adapter:
    global _MODEL
    if _MODEL is None:
        services.MlflowService().start()
        loader = registries.CustomLoader()
        model_uri = registries.uri_for_model_alias_or_version(
            name="agri", alias_or_version="Champion"
        )
        _MODEL = loader.load(uri=model_uri)
    return _MODEL
