"""Predict agricultural yields."""

import logging
import os
import warnings

# Suppress pandera import warning
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

# Suppress mlflow type hint warning
warnings.filterwarnings(
    "ignore", message=".*Type hint used in the model's predict function.*"
)

# Reduce MLflow verbosity
logging.getLogger("mlflow").setLevel(logging.WARNING)
