"""Tuned models and named ensembles used for the final submissions.

Hyperparameters come from the searches in scripts/evaluation (tune_model.py with GridSearchCV,
optuna_model.py with Optuna). The comments show the 5-fold stratified CV accuracy of the search
(seed 42 for the GridSearchCV ones, seed 23081999 for the Optuna ones).
"""

from .catalog import build_model
from .ensemble import stacking_ensemble, voting_ensemble


####################################################################################
# TUNED MODELS
####################################################################################

TUNED_MODELS = {
    # Found with GridSearchCV
    "gradient_boosting": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 2},  # 0.6828
    "logreg": {"C": 0.03, "penalty": "l1"},  # 0.6820
    "hist_gradient_boosting": {
        "max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 50,
    },  # 0.6809
    "gd_logreg": {"learning_rate": 0.1, "l2": 0.1},  # 0.6772
    "random_forest": {"min_samples_leaf": 1, "max_features": "sqrt"},  # 0.6739

    # Found with Optuna (50 trials each)
    "catboost": {"iterations": 1200, "learning_rate": 0.022, "depth": 6, "l2_leaf_reg": 3.53},  # 0.6830
    "xgboost": {
        "n_estimators": 650, "learning_rate": 0.0536, "max_depth": 2, "min_child_weight": 12,
        "subsample": 0.779, "colsample_bytree": 0.410, "reg_lambda": 0.0134,
    },  # 0.6825
}


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
