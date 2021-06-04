import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, ClassifierMixin
from neighbors.utils import ndcorrelate
from neighbors import metrics

eps = np.finfo(float).eps

# distance metrics
COSINE = 'cosine'
MINKOWSKI = 'minkowski'
EUCLIDEAN = 'minkowski2'
DISTANCE_METRICS = [COSINE, MINKOWSKI, EUCLIDEAN]

# voting options
MAJORITY = 'majority'
INVERSE_DISTANCE_WEIGHTED = 'idw'
SHEPARDS_WORK = 'sheppards'
VOTING = [MAJORITY, INVERSE_DISTANCE_WEIGHTED, SHEPARDS_WORK]

# weights
UNIFORM = 'uniform'
WEIGHTS = [UNIFORM]

# distance computation methods
SCIPY = 'scipy'
MAT = 'mat'
DISTANCE_METHODS = [SCIPY, MAT]


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5,
                 *, weights='uniform',
                 metric='minkowski',
                 voting='majority',
                 p=1,
                 method='scipy'):

        self.k = n_neighbors
        self.voting = voting
        self.weights = weights
        self.metric = metric
        self.p = p
        self.method = method
        self._validateParameters()

    def _computeFeatureWeights(self):
        if self.weights == UNIFORM:
            self.w = np.ones((self.trainX.shape[1],))
        self.w = self.w / self.w.max()

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        return self._predict(X)

    def _fit(self, X, y):
        assert X.shape[0] >= self.k, f"Need a minimum of {self.k} points"
        self.trainX = self._validate_data(X)
        self.trainTargets = y.copy()
        self._computeFeatureWeights()
        return self

    def _predict(self, X):
        X = self._validate_data(X)
        distanceMatrix = self.computeDistanceMatrix(X, self.trainX, self.w, self.metric, self.method)
        knnIndexes = self._computeKNNIndex(distanceMatrix)
        knnLabels = self._extractLabels(knnIndexes)
        return knnLabels

    def _extractLabels(self, knnIndexes):
        return self.trainTargets[knnIndexes]

    def _decide(self, knnLabels, distanceMatrix):
        if self.voting == MAJORITY:
            votingWeights = np.ones_like(knnLabels)
        elif self.voting == INVERSE_DISTANCE_WEIGHTED:
            votingWeights = 1 / (distanceMatrix[:, :self.k] + eps) ** self.p
        elif self.voting == SHEPARDS_WORK:
            votingWeights = np.exp(-1*distanceMatrix[:, :self.k])
        return self._computeDecision(knnLabels, votingWeights)

    def _computeDecision(self, knnLabels, votingWeights):
        numClasses = int(self.trainTargets.max()) + 1
        votes = np.empty((numClasses, *knnLabels.shape), dtype=int)
        for classNum in range(numClasses):
            votes[classNum] = np.where(knnLabels == classNum, 1, 0)
        weightedVotes = np.expand_dims(votingWeights, axis=0) * votes
        finalVotesPerClass = np.sum(weightedVotes, axis=2).T
        return np.argmax(finalVotesPerClass, axis=1)

    def _computeKNNIndex(self, distanceMatrix):
        return np.argsort(distanceMatrix)[:,:self.k]

    def computeDistanceMatrix(self, X, trainX, w, metric=MINKOWSKI, method=SCIPY):
        if method == MAT:
            return self._matricialDistanceMatrix(X, trainX, w, metric)
        elif method == SCIPY:
            return self._scipyDistanceMatrix(X, trainX, w, metric)

    @staticmethod
    def _scipyDistanceMatrix(X, trainX, w, metric):
        if metric == EUCLIDEAN:
            return cdist(X, trainX, metric=MINKOWSKI, p=2, w=w)
        elif metric == MINKOWSKI:
            return cdist(X, trainX, metric=MINKOWSKI, p=1, w=w)
        return cdist(X, trainX, metric=metric, w=w)

    @staticmethod
    def _matricialDistanceMatrix(X, trainX, w, metric):
        if metric == COSINE:
            return metrics.cosineDistance(X, trainX, w=w)
        elif metric == MINKOWSKI:
            return metrics.minkowskiDistance(X, trainX, w=w, p=1)
        elif metric == EUCLIDEAN:
            return metrics.euclideanDistance(X, trainX, w=w)

    def _validateParameters(self):
        assert self.k > 0, f"n_neighbors must be positive, not \'{self.k}\'"
        assert self.p > 0 and isinstance(self.p, int), f"p for distance voting must be a positive int"
        assert self.voting in VOTING, f"voting \'{self.voting}\' type not supported"
        assert self.weights in WEIGHTS, f"weights \'{self.weights}\' type not supported"
        assert self.metric in DISTANCE_METRICS, f"distance metric \'{self.metric}\' type not supported"
        assert self.method in DISTANCE_METHODS, f"distance computation method \'{self.method}\' not supported"

if __name__ == "__main__":
    N_features = 4
    X = np.random.rand(3, 4)
    y = np.eye(3) # one hot encoded labels
    clf = KNeighborsClassifier(1).fit(X, y)
    pred = clf.predict(X)[:,0,:]