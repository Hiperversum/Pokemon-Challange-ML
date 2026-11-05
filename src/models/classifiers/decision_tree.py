"""A single decision tree."""

from sklearn.tree import DecisionTreeClassifier

from ...utility.config.config_constants import RANDOM_STATE


####################################################################################
# DECISION TREE
####################################################################################

def decision_tree(max_depth=6, min_samples_leaf=20):
    """Single CART decision tree."""
    return DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=RANDOM_STATE
    )
