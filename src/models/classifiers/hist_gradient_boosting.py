"""Histogram gradient boosting: gradient boosting that groups the values of each feature into bins."""

from sklearn.ensemble import HistGradientBoostingClassifier

from ...utility.config.config_constants import RANDOM_STATE


####################################################################################
# HISTOGRAM GRADIENT BOOSTING
####################################################################################

def hist_gradient_boosting(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                           l2_regularization=1.0):
    """Histogram-based gradient boosting (faster on many samples)."""
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
        random_state=RANDOM_STATE,
    )
