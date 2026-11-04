"""Hyperparameter search with Optuna, on the same k-fold splits used everywhere."""

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from ..utility.config.config_constants import N_FOLDS, RANDOM_STATE
from ..models.catalog import build_model
from ..utility.print.messages import print_info, print_trial, timed_step
from .validation import make_folds


####################################################################################
# OPTUNA
####################################################################################

# For each model: a function that asks Optuna for one value of every hyperparameter.
# Optuna learns from the previous trials which values look promising (it does not try them all).
OPTUNA_SPACES = {
    "logreg": lambda trial: {
        "C": trial.suggest_float("C", 1e-4, 1.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
    },
    "gd_logreg": lambda trial: {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2": trial.suggest_float("l2", 0.001, 1.0, log=True),
    },
    "knn": lambda trial: {
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 200),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
    },
    "tree": lambda trial: {
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200),
    },
    "random_forest": lambda trial: {
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.02, 0.3, log=True),
    },
    "adaboost": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 1.5, log=True),
    },
    "gradient_boosting": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    },
    "xgboost": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 30.0, log=True),
    },
    "catboost": lambda trial: {
        "iterations": trial.suggest_int("iterations", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 2, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
    },
    "hist_gradient_boosting": lambda trial: {
        "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 30.0, log=True),
    },
}


def optuna_search(name, X, y, n_trials=50, n_folds=N_FOLDS):
    """Search the hyperparameters of one model with Optuna and return (best params, trials table).

    Every trial is scored with the same stratified k-fold used everywhere in the project.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """Score one set of hyperparameters with k-fold accuracy."""
        model = build_model(name, **OPTUNA_SPACES[name](trial))
        # The 5 folds are trained at the same time on different cores (n_jobs=-1)
        return cross_val_score(model, X, y, cv=make_folds(n_folds), scoring="accuracy", n_jobs=-1).mean()

    def report(study, trial):
        """Print one line after each trial."""
        print_trial(trial.number + 1, n_trials, trial.value, study.best_value)

    print_info("trials", f"{n_trials} x {n_folds} folds = {n_trials * n_folds} fits")
    study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
            )
    with timed_step(f"Optuna search for {name}"):
        study.optimize(objective, n_trials=n_trials, callbacks=[report])
    table = study.trials_dataframe().sort_values("value", ascending=False).reset_index(drop=True)
    return study.best_params, table
