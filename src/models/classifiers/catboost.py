"""CatBoost: the gradient boosting of the catboost library."""

from catboost import CatBoostClassifier

from ...utility.config.config_constants import RANDOM_STATE


####################################################################################
# CATBOOST
####################################################################################

def catboost(iterations=500, learning_rate=0.05, depth=4, l2_leaf_reg=3.0):
    """CatBoost: gradient boosting with symmetric trees."""
    return CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth, l2_leaf_reg=l2_leaf_reg,
        random_seed=RANDOM_STATE, verbose=0, thread_count=-1, allow_writing_files=False,
    )
