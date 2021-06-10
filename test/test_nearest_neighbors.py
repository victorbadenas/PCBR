import sys
import unittest
import numpy as np

sys.path.append('src/')

from neighbors.knn import KNeighborsClassifier


class TestNeighbors(unittest.TestCase):

    def test_neighbors(self):
        N_features = 4
        X = np.random.rand(3, 4)
        y = np.eye(3)  # one hot encoded labels
        clf = KNeighborsClassifier(1).fit(X, y)
        pred, distance_matrix = clf.predict(X)
        pred = pred[:, 0, :]
        self.assertTrue((pred == y).all())
        self.assertTrue(distance_matrix.shape == (3, 1))


if __name__ == "__main__":
    unittest.main()
