"""Hyperparameter search with Optuna, scored with stratified k-fold repeated several times."""

import optuna
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from ..utility.config.config_constants import N_FOLDS, RANDOM_STATE
from ..models.core.model_builder import build_model
from ..utility.print.messages import print_info, print_trial, timed_step

# Each trial is scored on the k folds drawn N_REPEATS times in different ways: with one fixed
# split Optuna could choose the values that work well only on those particular folds
N_REPEATS = 3


####################################################################################
# OPTUNA SPACES:
####################################################################################

# For each model: a function that asks Optuna for one value of every hyperparameter
OPTUNA_SPACES = {
    "logreg": lambda trial: {
        "C": trial.suggest_float("C", 1e-4, 1.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
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
    "xgboost": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 30.0, log=True),
    },
}


####################################################################################
# OPTUNA SEARCH: 
####################################################################################

def optuna_search(name, X, y, n_trials=50, n_folds=N_FOLDS):
    """Search the hyperparameters of one model with Optuna and return (best params, trials table).

    Every trial is scored with the stratified k-fold of the project, repeated N_REPEATS times
    with different folds (RepeatedStratifiedKFold).
    """
    # Remove the Optuna messages that are annoying when running many trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # The objective of the study: the score of one combination of hyperparameters
    def objective(trial):
        """Score one set of hyperparameters with repeated k-fold accuracy."""
        model = build_model(name, **OPTUNA_SPACES[name](trial))
        cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
        # The k x N_REPEATS fits are trained at the same time on different cores (n_jobs=-1)
        return cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()

    def report(study, trial):
        """Print one line after each trial."""
        print_trial(trial.number + 1, n_trials, trial.value, study.best_value)

    # Start the study
    print_info("trials", f"{n_trials} x {n_folds} folds x {N_REPEATS} repeats = "
                         f"{n_trials * n_folds * N_REPEATS} fits")
    study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
            )
    with timed_step(f"Optuna search for {name}"):
        study.optimize(objective, n_trials=n_trials, callbacks=[report])
    table = study.trials_dataframe().sort_values("value", ascending=False).reset_index(drop=True)
    return study.best_params, table
