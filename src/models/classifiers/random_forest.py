"""Random forest: many decision trees trained on random parts of the data (bagging)."""

from sklearn.ensemble import RandomForestClassifier

from ...utility.config.config_constants import RANDOM_STATE


####################################################################################
# RANDOM FOREST
####################################################################################

def random_forest(n_estimators=500, max_depth=None, min_samples_leaf=3, max_features="sqrt"):
    """Bagging of decorrelated trees."""
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        max_features=max_features, n_jobs=-1, random_state=RANDOM_STATE,
    )
