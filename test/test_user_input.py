import sys
import unittest
import numpy as np

sys.path.append('src/')

from user_request import UserRequest
from constraints import Constraints

class DummyUserRequest(UserRequest):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestUserRequest(unittest.TestCase):

    profile_str = "2, 1, 0, 1, 3, 1, 0, 0, 0, 1, 0, 0"
    preferences_str = "5, 2, 3, 1, 2, 1, 3, 4, 1, 0, 1, 0, 0"
    constraints_str = "cpu_brand: Intel, gpu_brand: PreferNVIDIA, max_budget: 1000"

    dur = DummyUserRequest(**{'profile_format':UserRequest.profile_format})

    def test_profile(self):
        result = UserRequest._process_profile(self.dur, self.profile_str)
        self.assertTrue(np.isclose(result, np.array([2., 1., 0., 1., 3., 1., 0., 0., 0., 1., 0., 0.])).all())

    def test_preferences(self):
        numitems = len(self.preferences_str.split(','))
        feature_relevance_matrix = np.ones((numitems, numitems)) / (numitems**2)
        feature_relevance_scaled, preferences_scaled = UserRequest._process_preferences(self.dur, self.preferences_str, feature_relevance_matrix)
        self.assertTrue(np.isclose(feature_relevance_scaled, np.full((13,), 0.51553254)).all())
        self.assertTrue((preferences_scaled == np.array([1.,0.25,0.5,0.,0.25,0.,0.5,0.75,1.,0.,1.,0.,0.])).all())

    def test_constraints(self):
        target = Constraints({'cpu_brand' : 'Intel', 'gpu_brand' : 'PreferNVIDIA', 'max_budget' : '1000'})
        result = UserRequest._process_constraints(self.dur, self.constraints_str)
        self.assertEqual(result.__dict__, target.__dict__)


if __name__ == "__main__":
    unittest.main()