"""Ensembles: several models combined into one (soft voting or stacking).

Both functions receive the members already built, as a list of (name, model) pairs.
"""

from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ...utility.config.config_constants import N_FOLDS, RANDOM_STATE


####################################################################################
# VOTING: the average of the members
####################################################################################

def voting_ensemble(estimators, voting="soft"):
    """Average the predicted probabilities (soft) or the votes (hard) of several models."""
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)


####################################################################################
# STACKING: a logistic regression learns how much each member counts
####################################################################################

def stacking_ensemble(estimators, final_C=1.0):
    """Train a logistic regression on the out-of-fold predictions of several models."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=final_C, max_iter=5000),
        cv=cv,
        n_jobs=-1,
    )
