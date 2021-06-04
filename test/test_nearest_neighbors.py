import sys
import unittest
import numpy as np

sys.path.append('src/')

from neighbors.knn import KNeighborsClassifier


class TestNeighbors(unittest.TestCase):

    def test_neighbors(self):
        N_features = 4
        X = np.random.rand(3, 4)
        y = np.eye(3) # one hot encoded labels
        clf = KNeighborsClassifier(1).fit(X, y)
        pred = clf.predict(X)[:,0,:]
        self.assertTrue((pred == y).all())

if __name__ == "__main__":
    unittest.main()
