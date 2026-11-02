"""Voting and stacking ensembles built from the base models."""

from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .. import config
from .catalog import build_model


def _named_models(model_specs):
    """Turn {"name": params} into a list of (name, estimator) pairs."""
    return [(name, build_model(name, **(params or {}))) for name, params in model_specs.items()]


def voting_ensemble(model_specs, voting="soft"):
    """Average the predicted probabilities (soft) or the votes (hard) of several models."""
    return VotingClassifier(estimators=_named_models(model_specs), voting=voting, n_jobs=-1)


def stacking_ensemble(model_specs, final_C=1.0):
    """Train a logistic regression on the out-of-fold predictions of several models."""
    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    return StackingClassifier(
        estimators=_named_models(model_specs),
        final_estimator=LogisticRegression(C=final_C, max_iter=5000),
        cv=cv,
        n_jobs=-1,
    )
