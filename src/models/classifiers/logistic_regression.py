"""Logistic regression on standardized features, and its version with polynomial features."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ..core.preprocessing import scaled


####################################################################################
# LOGISTIC REGRESSION
####################################################################################

def logistic_regression(C=0.1, penalty="l2", n_components=None):
    """Regularized logistic regression on standardized features."""
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return scaled(LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=5000), n_components)


####################################################################################
# WITH POLYNOMIAL FEATURES (polynomial experiment)
####################################################################################

def polynomial_logistic_regression(columns, degree=2, C=0.1, penalty="l2"):
    """Logistic regression with polynomial features (squares and products) of the given columns."""
    expand = ColumnTransformer(
        [("poly", PolynomialFeatures(degree=degree, include_bias=False), columns)],
        remainder="passthrough",
    )
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return Pipeline([
        ("expand", expand),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=5000)),
    ])
