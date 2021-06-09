"""
.. module:: KNeighborsClassifier
KNeighborsClassifier
*************
:Description: KNeighborsClassifier
    
:Version: 0.1.0
:Created on: 01/06/2021 11:00 
"""

__title__ = 'KNeighborsClassifier'
__version__ = '0.1.0'

import numpy as np
from . import metrics
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin

eps = np.finfo(float).eps

# distance metrics
COSINE = 'cosine'
MINKOWSKI = 'minkowski'
EUCLIDEAN = 'minkowski2'
DISTANCE_METRICS = [COSINE, MINKOWSKI, EUCLIDEAN]

# weights
UNIFORM = 'uniform'
WEIGHTS = [UNIFORM]

# distance computation methods
SCIPY = 'scipy'
MAT = 'mat'
DISTANCE_METHODS = [SCIPY, MAT]


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    __version__ = '0.1.0'
    __doc__ = ""

    def __init__(self, n_neighbors=5,
                 *, weights='uniform',
                 metric='minkowski',
                 method='scipy'):
        """KNeighborsClassifier object

        Args:
            n_neighbors (int, optional): [ number of neighbors to extract]. 
                Defaults to 5.
            weights (str, optional): [options for computing the weight of 
                each of the features. Not supported right now]. Defaults to 'uniform'.
            metric (str, optional): [distance used to compute the distance matrixes. 
                Available options are: "cosine", "minkowski" or "euclidean"]. Defaults to 'minkowski'.
            method (str, optional): [method for computing the distance matrix. 
                'scipy' uses scipy.distance.cdist to compute the distances, 
                which is more efficient in memory. 'mat' computes the distances faster 
                but uses more memory]. Defaults to 'scipy'.
        """
        self.k = n_neighbors
        self.weights = weights
        self.metric = metric
        self.method = method
        self._validateParameters()

    def _computeFeatureWeights(self):
        """computes the weight for each one of the features.
        currenly not supported and it defaults to uniform.
        """
        if self.weights == UNIFORM:
            self.w = np.ones((self.trainX.shape[1],))
        elif isinstance(self.weights, np.ndarray):
            assert self.weights.shape[0] == self.trainX.shape[1], f"feature mismatch {self.weights.shape[0]} != {self.trainX.shape[1]}"
            self.w = self.weights
        self.w = self.w / self.w.sum()

    def fit(self, X, y):
        """fit object to training data

        Args:
            X ([np.ndarray, pd.DataFrame]): [training data of shape (n_samples, n_origin_features)]
            y ([np.ndarray]): [array of shape (n_samples, n_target_features) or (n_samples,) 
            if the classification returns an integer value.]

        Returns:
            [KNeighborsClassifier]: [fitted classifier]
        """
        return self._fit(X, y)

    def predict(self, X):
        """inference function to assign the closest point in the training data to each instance in X

        Args:
            X ([np.ndarray]): [of shape (n_samples, n_origin_features)]

        Returns:
            [np.ndarray]: [of shape (n_samples, n_neighbors, n_target_features)]
        """
        return self._predict(X)

    def _fit(self, X, y):
        """internal fit method. Validates the data, computes feature weights and stores the training data in memory.

        Args:
            X ([np.ndarray, pd.DataFrame]): [training data of shape (n_samples, n_origin_features)]
            y ([np.ndarray]): [array of shape (n_samples, n_target_features) or (n_samples,) 
            if the classification returns an integer value.]

        Returns:
            [KNeighborsClassifier]: [fitted classifier]
        """
        assert X.shape[0] >= self.k, f"Need a minimum of {self.k} points"
        self.trainX = self._validate_data(X)
        self.trainTargets = y.copy()
        self._computeFeatureWeights()
        return self

    def _predict(self, X):
        """internal inference method to assign the closest point in the training data to each instance in X

        Args:
            X ([np.ndarray]): [of shape (n_samples, n_origin_features)]

        Returns:
            [np.ndarray]: [of shape (n_samples, n_neighbors, n_target_features)]
        """
        X = self._validate_data(X)
        distanceMatrix = self.computeDistanceMatrix(X, self.trainX, self.w, self.metric, self.method)
        knnIndexes = self._computeKNNIndex(distanceMatrix)
        knnLabels = self._extractLabels(knnIndexes)
        return knnLabels, np.sort(distanceMatrix)[:, :self.k]

    def _extractLabels(self, knnIndexes):
        """maps the best indexes to the labels that correspond to those instances.
        """
        return self.trainTargets[knnIndexes]

    def _computeKNNIndex(self, distanceMatrix):
        """extracts the indexes for the best k elements in the training data.
        """
        return np.argsort(distanceMatrix)[:, :self.k]

    def computeDistanceMatrix(self, X, trainX, w, metric=MINKOWSKI, method=SCIPY):
        """computes the distance matrix. Switches between matricial or cdist operations.
        """
        if method == MAT:
            return self._matricialDistanceMatrix(X, trainX, w, metric)
        elif method == SCIPY:
            return self._scipyDistanceMatrix(X, trainX, w, metric)

    @staticmethod
    def _scipyDistanceMatrix(X, trainX, w, metric):
        """generates instance matrixes for cdist option. switches between distance types.
        """
        if metric == EUCLIDEAN:
            return cdist(X, trainX, metric=MINKOWSKI, p=2, w=w)
        elif metric == MINKOWSKI:
            return cdist(X, trainX, metric=MINKOWSKI, p=1, w=w)
        return cdist(X, trainX, metric=metric, w=w)

    @staticmethod
    def _matricialDistanceMatrix(X, trainX, w, metric):
        """generates instance matrixes for matricial option. switches between distance types.
        """
        if metric == COSINE:
            return metrics.cosineDistance(X, trainX, w=w)
        elif metric == MINKOWSKI:
            return metrics.minkowskiDistance(X, trainX, w=w, p=1)
        elif metric == EUCLIDEAN:
            return metrics.euclideanDistance(X, trainX, w=w)

    def _validateParameters(self):
        assert self.k > 0, f"n_neighbors must be positive, not \'{self.k}\'"
        assert self.weights in WEIGHTS or isinstance(self.weights, np.ndarray), f"weights \'{self.weights}\' type not supported"
        assert self.metric in DISTANCE_METRICS, f"distance metric \'{self.metric}\' type not supported"
        assert self.method in DISTANCE_METHODS, f"distance computation method \'{self.method}\' not supported"


if __name__ == "__main__":
    N_features = 4
    X = np.random.rand(3, 4)
    y = np.eye(3)  # one hot encoded labels
    clf = KNeighborsClassifier(1).fit(X, y)
    pred = clf.predict(X)[:, 0, :]
