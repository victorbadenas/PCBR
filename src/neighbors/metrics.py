import numpy as np

def cosineDistance(X, Y, w=None):
    if w is None:
        w = np.ones(X.shape[1])
    assert X.shape[1] == Y.shape[1] == w.shape[0]
    Y = Y * np.sqrt(w)[None, :]
    X = X * np.sqrt(w)[None, :]
    d = X@Y.T / np.linalg.norm(Y, axis=1)[None, :] / np.linalg.norm(X, axis=1, keepdims=True)
    return 1-d

def minkowskiDistance(X, Y, w=None, p=1):
    if w is None:
        w = np.ones(X.shape[1])
    assert X.shape[1] == Y.shape[1] == w.shape[0]
    d = np.expand_dims(Y, 1) - np.expand_dims(X, 0)
    d = np.sum((w[None, None, :] * np.abs(d)**p), axis=2)**(1/p)
    return d.T

def euclideanDistance(X, Y, w=None):
    if w is None:
        w = np.ones(X.shape[1])
    assert X.shape[1] == Y.shape[1] == w.shape[0]
    d = np.expand_dims(Y, 1) - np.expand_dims(X, 0)
    d = np.sqrt(np.sum(w[None, None, :] * d ** 2, axis=2))
    return d.T
