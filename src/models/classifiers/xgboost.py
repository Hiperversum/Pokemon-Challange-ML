"""XGBoost: the gradient boosting of the xgboost library."""

from xgboost import XGBClassifier

from ...utility.config.config_constants import RANDOM_STATE


####################################################################################
# XGBOOST
####################################################################################

def xgboost(n_estimators=400, learning_rate=0.05, max_depth=3, min_child_weight=1,
            subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0):
    """XGBoost: gradient boosting with depth-wise trees."""
    return XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda, tree_method="hist", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
