"""AdaBoost: small trees trained one after the other, each one focused on the previous mistakes."""

from sklearn.ensemble import AdaBoostClassifier

from ...utility.config.config_constants import RANDOM_STATE


####################################################################################
# ADABOOST
####################################################################################

def adaboost(n_estimators=300, learning_rate=0.5):
    """AdaBoost with decision stumps."""
    return AdaBoostClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, random_state=RANDOM_STATE
    )
