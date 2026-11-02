"""Hyperparameter search on the same k-fold splits: GridSearchCV and Optuna."""

import optuna
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score

from .. import config
from ..models.catalog import build_model
from ..utility.print.messages import print_info, print_trial, timed_step
from .validation import make_folds


####################################################################################
# GRID SEARCH
####################################################################################

# Scaled models are pipelines, so their parameters carry the "clf__" prefix.
PARAM_GRIDS = {
    "logreg": {
        "clf__C": [0.0003, 0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
    },
    "gd_logreg": {"clf__l2": [0.03, 0.1, 0.3, 1.0], "clf__learning_rate": [0.1]},
    "knn": {"clf__n_neighbors": [15, 31, 51, 101, 151], "clf__weights": ["uniform", "distance"]},
    "tree": {"max_depth": [3, 4, 6, 8, 10], "min_samples_leaf": [10, 30, 60, 100]},
    "random_forest": {"min_samples_leaf": [1, 3, 5, 10], "max_features": ["sqrt", 0.2, 0.4]},
    "adaboost": {"n_estimators": [200, 500], "learning_rate": [0.1, 0.5, 1.0]},
    "gradient_boosting": {
        "n_estimators": [200, 400, 800], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [1, 2, 3, 4],
    },
    "hist_gradient_boosting": {
        "max_iter": [200, 400], "learning_rate": [0.03, 0.05], "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [20, 50],
    },
}


def grid_search(name, X, y, param_grid=None, n_folds=config.N_FOLDS):
    """Run GridSearchCV on one model and return (best params, results table)."""
    param_grid = param_grid or PARAM_GRIDS[name]
    n_combinations = len(ParameterGrid(param_grid))
    print_info("combinations", n_combinations)
    print_info("fits", f"{n_combinations} x {n_folds} folds = {n_combinations * n_folds}")
    search = GridSearchCV(
        estimator=build_model(name),
        param_grid=param_grid,
        cv=make_folds(n_folds),
        scoring="accuracy",
        n_jobs=-1,
        refit=False,
        return_train_score=True,
    )
    with timed_step(f"Grid search for {name}"):
        search.fit(X, y)
    table = pd.DataFrame(search.cv_results_)[
        ["params", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score").reset_index(drop=True)
    return search.best_params_, table


####################################################################################
# OPTUNA
####################################################################################

# For each model: a function that asks Optuna for one value of every hyperparameter.
# Optuna learns from the previous trials which values look promising (it does not try them all).
OPTUNA_SPACES = {
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
    "logreg": lambda trial: {
        "C": trial.suggest_float("C", 1e-4, 1.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
    },
    "hist_gradient_boosting": lambda trial: {
        "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 30.0, log=True),
    },
}


def optuna_search(name, X, y, n_trials=50, n_folds=config.N_FOLDS):
    """Search the hyperparameters of one model with Optuna and return (best params, trials table).

    Every trial is scored with the same stratified k-fold used everywhere in the project.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """Score one set of hyperparameters with k-fold accuracy."""
        model = build_model(name, **OPTUNA_SPACES[name](trial))
        return cross_val_score(model, X, y, cv=make_folds(n_folds), scoring="accuracy").mean()

    def report(study, trial):
        """Print one line after each trial."""
        print_trial(trial.number + 1, n_trials, trial.value, study.best_value)

    print_info("trials", f"{n_trials} x {n_folds} folds = {n_trials * n_folds} fits")
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    )
    with timed_step(f"Optuna search for {name}"):
        study.optimize(objective, n_trials=n_trials, callbacks=[report])
    table = study.trials_dataframe().sort_values("value", ascending=False).reset_index(drop=True)
    return study.best_params, table
