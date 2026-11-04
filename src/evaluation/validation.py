"""Stratified k-fold cross-validation helpers."""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from ..utility.config.config_constants import N_FOLDS, RANDOM_STATE
from ..utility.print.messages import print_fold_done, print_model_result, print_model_start


def make_folds(n_folds=N_FOLDS, seed=RANDOM_STATE):
    """Return the shuffled stratified splitter used everywhere in the project."""
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)


def cross_validate(model, X, y, n_folds=N_FOLDS, seed=RANDOM_STATE, on_fold=None):
    """Fit the model on each fold and return train/validation accuracies.

    If given, on_fold(score) is called after each fold with its validation accuracy.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    train_scores, val_scores = [], []

    for train_idx, val_idx in make_folds(n_folds, seed).split(X, y):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])
        train_scores.append(accuracy_score(y[train_idx], fold_model.predict(X[train_idx])))
        val_pred = fold_model.predict(X[val_idx])
        val_scores.append(accuracy_score(y[val_idx], val_pred))
        if on_fold:
            on_fold(val_scores[-1])

    return {
        "train_acc": float(np.mean(train_scores)),
        "val_acc": float(np.mean(val_scores)),
        "val_std": float(np.std(val_scores)),
    }


def compare_models(models, X, y, n_folds=N_FOLDS, verbose=True):
    """Cross-validate several models and return a table sorted by validation accuracy."""
    rows = []
    for position, (name, model) in enumerate(models.items(), start=1):
        if verbose:
            print_model_start(position, len(models), name)
        start = time.time()
        result = cross_validate(model, X, y, n_folds, on_fold=print_fold_done if verbose else None)
        rows.append({
            "model": name,
            "train_acc": result["train_acc"],
            "val_acc": result["val_acc"],
            "val_std": result["val_std"],
        })
        if verbose:
            print_model_result(result, time.time() - start)
    return pd.DataFrame(rows).sort_values("val_acc", ascending=False).reset_index(drop=True)
