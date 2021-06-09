import sys
import unittest
import numpy as np

sys.path.append('src/')

from pcbr import UserRequest
from constraints import Constraints


class TestUserRequest(unittest.TestCase):

    profile_str = "2, 1, Programming, 1, 3, 1, 0, 0, 0, 1, 0, 0"
    preferences_str = "5, 2, 3, 1, 2, 1, 3, 4"
    constraints_str = "cpu_brand: Intel, gpu_brand: PreferNVIDIA, max_budget: 1000"

    def test_profile(self):
        result = UserRequest._process_profile(None, self.profile_str)
        self.assertEqual(result, None)

    def test_preferences(self):
        result = UserRequest._process_preferences(None, self.preferences_str)
        self.assertTrue((result == np.array([5, 2, 3, 1, 2, 1, 3, 4])).all())

    def test_constraints(self):
        target = Constraints({'cpu_brand' : 'Intel', 'gpu_brand' : 'PreferNVIDIA', 'max_budget' : '1000'})
        result = UserRequest._process_constraints(None, self.constraints_str)
        self.assertEqual(result.__dict__, target.__dict__)


if __name__ == "__main__":
    unittest.main()