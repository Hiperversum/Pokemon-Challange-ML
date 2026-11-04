"""Tuned models and named ensembles used for the final submissions.

The hyperparameters of the tuned models are read from data/models/tuned_params.json,
which the optuna command (scripts/evaluation/optuna_model.py) updates by itself.
"""

from ..utility.model.load_params import load_tuned_models
from .catalog import build_model
from .ensemble import stacking_ensemble, voting_ensemble


####################################################################################
# TUNED MODELS
####################################################################################

# {model: hyperparameters}, read from tuned_params.json
TUNED_MODELS = load_tuned_models()


def ensemble_members(*names):
    """Pick the tuned settings of the given models."""
    return {name: TUNED_MODELS[name] for name in names}


def build_tuned_model(name):
    """Create a single model with its tuned hyperparameters (defaults if it was not tuned)."""
    return build_model(name, **TUNED_MODELS.get(name, {}))


####################################################################################
# ENSEMBLES
####################################################################################

BEST_MODELS = ("logreg", "catboost", "xgboost", "gradient_boosting", "hist_gradient_boosting")

ENSEMBLES = {
    "voting": lambda: voting_ensemble(
        ensemble_members("logreg", "gradient_boosting", "hist_gradient_boosting"), voting="soft"),
    "voting_all": lambda: voting_ensemble(
        ensemble_members("logreg", "gradient_boosting", "hist_gradient_boosting",
                         "gd_logreg", "random_forest"), voting="soft"),
    "stacking": lambda: stacking_ensemble(
        ensemble_members("logreg", "gradient_boosting", "hist_gradient_boosting", "random_forest")),

    # With the Optuna models
    "voting_best": lambda: voting_ensemble(ensemble_members(*BEST_MODELS), voting="soft"),
    "stacking_best": lambda: stacking_ensemble(ensemble_members(*BEST_MODELS)),
}


def build_ensemble(name):
    """Create one of the named ensembles."""
    return ENSEMBLES[name]()
