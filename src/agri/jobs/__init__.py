"""High-level jobs of the project."""

# %% IMPORTS

from agri.jobs.evaluations import EvaluationsJob
from agri.jobs.explanations import ExplanationsJob
from agri.jobs.inference import InferenceJob
from agri.jobs.promotion import PromotionJob
from agri.jobs.training import TrainingJob
from agri.jobs.tuning import TuningJob

# %% TYPES

JobKind = (
    TuningJob
    | TrainingJob
    | PromotionJob
    | InferenceJob
    | EvaluationsJob
    | ExplanationsJob
)

# %% EXPORTS

__all__ = [
    "EvaluationsJob",
    "ExplanationsJob",
    "InferenceJob",
    "JobKind",
    "PromotionJob",
    "TrainingJob",
    "TuningJob",
]
