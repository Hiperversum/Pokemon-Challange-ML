"""Build every model of the project from its name.

- the catalog: name -> function that builds the model (each model has its own file in classifiers/);
- the tuned models: their hyperparameters are read from data/models/tuned_params.json;
- the ensembles: their type and members are read from data/models/ensembles.json.
"""

from ...utility.model.load_params import load_ensembles, load_tuned_models
from ..classifiers.adaboost import adaboost
from ..classifiers.catboost import catboost
from ..classifiers.decision_tree import decision_tree
from ..classifiers.ensemble import stacking_ensemble, voting_ensemble
from ..classifiers.gd_logistic_regression import gd_logistic_regression
from ..classifiers.gradient_boosting import gradient_boosting
from ..classifiers.hist_gradient_boosting import hist_gradient_boosting
from ..classifiers.knn import knn
from ..classifiers.logistic_regression import logistic_regression
from ..classifiers.random_forest import random_forest
from ..classifiers.xgboost import xgboost


####################################################################################
# CATALOG: the name used in config.json5 and in the json files -> the function that builds the model
####################################################################################

MODEL_FACTORIES = {
    "logreg": logistic_regression,
    "gd_logreg": gd_logistic_regression,
    "knn": knn,
    "tree": decision_tree,
    "random_forest": random_forest,
    "adaboost": adaboost,
    "gradient_boosting": gradient_boosting,
    "hist_gradient_boosting": hist_gradient_boosting,
    "xgboost": xgboost,
    "catboost": catboost,
}


####################################################################################
# BUILD A MODEL BY NAME (with the default hyperparameters, or the ones passed)
####################################################################################

def build_model(name, **params):
    """Create a model by name, overriding its default hyperparameters."""
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_FACTORIES)}")
    return MODEL_FACTORIES[name](**params)


####################################################################################
# TUNED MODELS (data/models/tuned_params.json)
####################################################################################

# {model: hyperparameters}, written by the tune command
TUNED_MODELS = load_tuned_models()


def build_tuned_model(name):
    """Create a single model with its tuned hyperparameters (defaults if it was not tuned)."""
    return build_model(name, **TUNED_MODELS.get(name, {}))


####################################################################################
# ENSEMBLES (data/models/ensembles.json)
####################################################################################

# {ensemble: {"type": "voting" or "stacking", "models": [names of the members]}}
ENSEMBLES = load_ensembles()


def build_ensemble(name):
    """Create one of the ensembles of ensembles.json, each member with its tuned hyperparameters."""
    spec = ENSEMBLES[name]

    # The members, as (name, model) pairs, each one with its tuned hyperparameters
    members = [(member, build_model(member, **TUNED_MODELS[member])) for member in spec["models"]]

    # Soft voting averages the probabilities; stacking weighs them with a logistic regression
    if spec["type"] == "voting":
        return voting_ensemble(members, voting="soft")
    return stacking_ensemble(members)
