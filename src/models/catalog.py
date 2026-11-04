"""All the classifiers of the project, each one built by a small function.

Models that depend on distances or on the scale of the weights
(logistic regression, kNN) get a StandardScaler in front, optionally
followed by a PCA that compresses the features into fewer components.
"""

from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .. import config
from .gradient_descent import GradientDescentLogisticRegression


####################################################################################
# PREPROCESSING
####################################################################################

def scaled(estimator, n_components=None):
    """Put a StandardScaler (and a PCA if n_components is given) in front of an estimator."""
    steps = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=config.RANDOM_STATE)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def pca_on_columns(estimator, columns, n_components):
    """Compress only the given columns with PCA, keep the other ones, then scale everything."""
    compress = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=config.RANDOM_STATE)),
    ])
    reduce = ColumnTransformer([("pca", compress, columns)], remainder="passthrough")
    return Pipeline([("reduce", reduce), ("scaler", StandardScaler()), ("clf", estimator)])


####################################################################################
# LINEAR AND DISTANCE MODELS
####################################################################################

def logistic_regression(C=0.1, penalty="l2", n_components=None):
    """Regularized logistic regression on standardized features."""
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return scaled(LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=5000), n_components)


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


def logistic_regression_pca_on(columns, n_components=20, C=0.02, penalty="l1"):
    """Logistic regression where only the given columns are compressed with PCA first.

    Used on the bag-of-tokens counts: hundreds of sparse columns become a few components,
    while the dense battle features (HP, KOs, statuses...) are kept as they are.
    """
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    estimator = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=5000)
    return pca_on_columns(estimator, columns, n_components)


def gd_logistic_regression(learning_rate=0.1, n_epochs=300, batch_size=None, l2=0.01):
    """Logistic regression trained with our gradient descent on standardized features."""
    return scaled(GradientDescentLogisticRegression(
        learning_rate=learning_rate, n_epochs=n_epochs, batch_size=batch_size, l2=l2,
        random_state=config.RANDOM_STATE,
    ))


def knn(n_neighbors=50, weights="distance", n_components=None):
    """k-Nearest Neighbors on standardized features."""
    return scaled(KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights), n_components)


####################################################################################
# TREES AND SKLEARN ENSEMBLES
####################################################################################

def decision_tree(max_depth=6, min_samples_leaf=20):
    """Single CART decision tree."""
    return DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=config.RANDOM_STATE
    )


def random_forest(n_estimators=500, max_depth=None, min_samples_leaf=3, max_features="sqrt"):
    """Bagging of decorrelated trees."""
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        max_features=max_features, n_jobs=-1, random_state=config.RANDOM_STATE,
    )


def adaboost(n_estimators=300, learning_rate=0.5):
    """AdaBoost with decision stumps."""
    return AdaBoostClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, random_state=config.RANDOM_STATE
    )


def gradient_boosting(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8):
    """Classic gradient boosted trees."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=subsample, random_state=config.RANDOM_STATE,
    )


def hist_gradient_boosting(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                           l2_regularization=1.0):
    """Histogram-based gradient boosting (faster on many samples)."""
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
        random_state=config.RANDOM_STATE,
    )


####################################################################################
# GRADIENT BOOSTING LIBRARIES
####################################################################################

def xgboost(n_estimators=400, learning_rate=0.05, max_depth=3, min_child_weight=1,
            subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0):
    """XGBoost: gradient boosting with depth-wise trees."""
    return XGBClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda, tree_method="hist", eval_metric="logloss",
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )


def catboost(iterations=500, learning_rate=0.05, depth=4, l2_leaf_reg=3.0):
    """CatBoost: gradient boosting with symmetric trees."""
    return CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth, l2_leaf_reg=l2_leaf_reg,
        random_seed=config.RANDOM_STATE, verbose=0, thread_count=-1, allow_writing_files=False,
    )


####################################################################################
# CATALOG
####################################################################################

MODEL_FACTORIES = {
    "logreg": logistic_regression,
    "gd_logreg": gd_logistic_regression,
    "knn": knn,
    "tree": decision_tree,
    "random_forest": random_forest,
    "adaboost": adaboost,
    "gradient_boosting": gradient_boosting,
    "hist_gradient_boosting": hist_gradient_boosting,
    "xgboost": xgboost,
    "catboost": catboost,
}


def build_model(name, **params):
    """Create a model by name, overriding its default hyperparameters."""
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_FACTORIES)}")
    return MODEL_FACTORIES[name](**params)
