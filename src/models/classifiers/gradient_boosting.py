"""Gradient boosting: trees trained one after the other, each one correcting the previous ones."""

from sklearn.ensemble import GradientBoostingClassifier

from ...utility.config.config_constants import RANDOM_STATE


####################################################################################
# GRADIENT BOOSTING
####################################################################################

def gradient_boosting(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8):
    """Classic gradient boosted trees."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=subsample, random_state=RANDOM_STATE,
    )
