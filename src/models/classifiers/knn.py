"""k-Nearest Neighbors: a battle gets the result of the most similar training battles."""

from sklearn.neighbors import KNeighborsClassifier

from ..core.preprocessing import scaled


####################################################################################
# K-NEAREST NEIGHBORS
####################################################################################

def knn(n_neighbors=50, weights="distance", n_components=None):
    """k-Nearest Neighbors on standardized features."""
    return scaled(KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights), n_components)
