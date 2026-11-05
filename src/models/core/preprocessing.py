"""Steps put in front of a model: standardization, and PCA when it is asked for."""

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ...utility.config.config_constants import RANDOM_STATE


####################################################################################
# STANDARDIZATION (AND PCA)
####################################################################################

def scaled(estimator, n_components=None):
    """Put a StandardScaler (and a PCA if n_components is given) in front of an estimator."""
    steps = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)))
    steps.append(("clf", estimator))
    return Pipeline(steps)


####################################################################################
# PCA ON SOME COLUMNS ONLY (pca experiment)
####################################################################################

def pca_on_columns(estimator, columns, n_components):
    """Compress only the given columns with PCA, keep the other ones, then scale everything."""
    compress = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
    ])
    reduce = ColumnTransformer([("pca", compress, columns)], remainder="passthrough")
    return Pipeline([("reduce", reduce), ("scaler", StandardScaler()), ("clf", estimator)])
