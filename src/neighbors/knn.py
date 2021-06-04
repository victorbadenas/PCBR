import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, ClassifierMixin
# from neighbors.utils import ndcorrelate
# from neighbors import metrics
from utils import ndcorrelate
import metrics

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
MUTUAL_INFO = 'mutual_info'
CORRELATION = "correlation"
WEIGHTS = [UNIFORM, MUTUAL_INFO, CORRELATION]

# distance computation methods
SCIPY = 'scipy'
MAT = 'mat'
DISTANCE_METHODS = [SCIPY, MAT]


class kNNAlgorithm(BaseEstimator, ClassifierMixin):
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
        elif self.weights == MUTUAL_INFO:
            self.w = mutual_info_classif(self.trainX, self.trainLabels)
        elif self.weights == CORRELATION:
            self.w = ndcorrelate(self.trainX, self.trainLabels)
            self.w[self.w < 0] = 0
            if np.sum(self.w) == 0:
                print("Correlation weights sum 0, defaulting to uniform weights")
                self.w = np.ones((self.trainX.shape[1],))
        self.w = self.w / self.w.max()

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        return self._predict(X)

    def _fit(self, X, y):
        assert X.shape[0] >= self.k, f"Need a minimum of {self.k} points"
        self.trainX, self.trainLabels = self._validate_data(X, y)
        self._computeFeatureWeights()
        return self

    def _predict(self, X):
        X = self._validate_data(X)
        distanceMatrix = self.computeDistanceMatrix(X, self.trainX, self.w, self.metric, self.method)
        knnIndexes = self._computeKNNIndex(distanceMatrix)
        knnLabels = self._extractLabels(knnIndexes)
        decision = self._decide(knnLabels, distanceMatrix)
        return decision

    def _extractLabels(self, knnIndexes):
        labels = self.trainLabels[knnIndexes]
        return labels.astype(np.int)

    def _decide(self, knnLabels, distanceMatrix):
        if self.voting == MAJORITY:
            votingWeights = np.ones_like(knnLabels)
        elif self.voting == INVERSE_DISTANCE_WEIGHTED:
            votingWeights = 1 / (distanceMatrix[:, :self.k] + eps) ** self.p
        elif self.voting == SHEPARDS_WORK:
            votingWeights = np.exp(-1*distanceMatrix[:, :self.k])
        return self._computeDecision(knnLabels, votingWeights)

    def _computeDecision(self, knnLabels, votingWeights):
        numClasses = int(self.trainLabels.max()) + 1
        votes = np.empty((numClasses, *knnLabels.shape), dtype=int)
        for classNum in range(numClasses):
            votes[classNum] = np.where(knnLabels == classNum, 1, 0)
        weightedVotes = np.expand_dims(votingWeights, axis=0) * votes
        finalVotesPerClass = np.sum(weightedVotes, axis=2).T
        return np.argmax(finalVotesPerClass, axis=1)

    def _computeKNNIndex(self, distanceMatrix):
        knnIndex = [None]*distanceMatrix.shape[0]
        for i in range(distanceMatrix.shape[0]):
            knnIndex[i] = np.argsort(distanceMatrix[i, :])[:self.k]
        return np.vstack(knnIndex)

    @staticmethod
    def computeDistanceMatrix(X, trainX, w, metric=MINKOWSKI, method=MAT):
        if method == MAT:
            return kNNAlgorithm._matricialDistanceMatrix(X, trainX, w, metric)
        elif method == SCIPY:
            return kNNAlgorithm._scipyDistanceMatrix(X, trainX, w, metric)

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
    import matplotlib.pyplot as plt
    from pathlib import Path

    N_features = 2
    classgs = np.array([
        ((0.75, 0.75) + tuple([0.75]*(N_features-2))),
        ((0.25, 0.25) + tuple([0.75]*(N_features-2))),
        ((0.75, 0.25) + tuple([0.75]*(N_features-2))),
        ((0.25, 0.75) + tuple([0.75]*(N_features-2)))
    ])

    data = []
    labels = []
    data.append(np.random.rand(50, N_features)/2 - 0.25 + ((0.75, 0.75) + tuple([0.75]*(N_features-2))))
    labels.append(np.zeros((50,)))
    data.append(np.random.rand(50, N_features)/2 - 0.25 + ((0.25, 0.25) + tuple([0.75]*(N_features-2))))
    labels.append(np.full((50,), 1))
    data.append(np.random.rand(50, N_features)/2 - 0.25 + ((0.75, 0.25) + tuple([0.75]*(N_features-2))))
    labels.append(np.full((50,), 2))
    data.append(np.random.rand(50, N_features)/2 - 0.25 + ((0.25, 0.75) + tuple([0.75]*(N_features-2))))
    labels.append(np.full((50,), 3))
    data = np.vstack(data)
    labels = np.concatenate(labels)

    newData = np.random.rand(50, N_features)
    newLabels = np.argmin(cdist(newData, classgs), axis=1)

    def plotModelTrial(trainData, testData, trainLabels, testLabels, classgs):
        plt.figure(figsize=(15, 9))
        for label, c in zip(np.unique(trainLabels), 'rgby'):
            subData = trainData[trainLabels == label]
            subNewData = testData[testLabels == label]
            plt.scatter(subData[:, 0], subData[:, 1], c=c, marker='+')
            plt.scatter(subNewData[:, 0], subNewData[:, 1], c=c, marker='x')
        # plt.scatter(classgs[:, 0], classgs[:, 1], c='k', marker='o')
        plt.vlines(0.5, 0, 1, colors='k', linestyles='dashed')
        plt.hlines(0.5, 0, 1, colors='k', linestyles='dashed')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([i/4 for i in range(5)])
        plt.yticks([i/4 for i in range(5)])
        plt.grid('on')

    plotModelTrial(data, newData, labels, newLabels, classgs)
    plt.show()

    print(f"train dataset size: {data.shape}, test dataset size: {newData.shape}")
    for d in DISTANCE_METRICS:
        for v in VOTING:
            for w in WEIGHTS:
                for m in [MAT]:
                    print(f"distance: {d}, voting: {v}, weights: {w}, method {m}")
                    knn = kNNAlgorithm(metric=d, voting=v, weights=w, method=m)
                    pred_labels = knn.fit(data, labels).predict(newData)
                    print(pred_labels)
