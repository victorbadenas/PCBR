import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix


class NearestNeighbors:
    def __init__(self, n_neighbors=3, metric='minkowski'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._reset()

    def _reset(self):
        self.fitted = False
        self.train_x_ = None

    def fit(self, X, y=None):
        return self._fit(X)

    def _fit(self, X):
        self.n_fit_samples_, self.n_dimensions_in_ = X.shape[:2]

        self.train_x_ = X

        if self.n_fit_samples_ < self.n_neighbors:
            msg = f"number of instances in fit data must be bigger than the n_neighbors. Instances in fit data: {self.n_fit_samples_}, n_neighbors: {self.n_neighbors}"
            raise ValueError(msg)

        self.fitted = True
        return self

    def check_fitted(self):
        if not self.fitted:
            raise ValueError(f"{self.__class__.__name__} has not been fitted")

    def kneighbors(self, X=None, return_distance=True):
        self.check_fitted()
        if X is None:
            X = self.train_x_.copy()
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        n_instances, n_features = X.shape[:2]

        if self.n_dimensions_in_ != n_features:
            msg = f'incoherent number of dimensions {self.n_dimensions_in_} != {n_features}'
            raise ValueError(msg)

        distances = cdist(X, self.train_x_, metric=self.metric)
        sorted_indices = np.argsort(distances, axis=-1)[:, :self.n_neighbors]

        if return_distance:
            sorted_distances = np.sort(distances, axis=-1)[:, :self.n_neighbors]
            return sorted_distances, sorted_indices
        else:
            return sorted_indices

    def kneighbors_graph(self, X=None):
        self.check_fitted()
        if X is None:
            X = self.train_x_.copy()

        # get knn index and distances
        distances, indexes = self.kneighbors(X, return_distance=True)

        # flatten
        distances = np.ravel(distances)

        # index pointers
        ind_ptr = np.arange(0, indexes.shape[0] * self.n_neighbors + 1, self.n_neighbors)

        return csr_matrix((distances, np.ravel(indexes), ind_ptr), shape=(indexes.shape[0], self.n_fit_samples_))
