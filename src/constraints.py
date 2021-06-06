import unittest

class Constraints:
    def __init__(self, constraint_dict={}):
        """initialize the constraints structure using a dictionary of keys/values from user input
        """

        # Initialize everything to default (no constraints)
        self.cpu_brand=None
        self.gpu_brand=None
        self.min_ram=None
        self.max_budget=None

        # Potential for Future: If we want to specify specific CPU/GPU
        #self.cpu=None
        #self.gpu=None

        for k in constraint_dict:
            if k=='cpu_brand':
                if constraint_dict[k] in ['Intel', 'PreferIntel', 'PreferAMD', 'AMD']:
                    self.cpu_brand = constraint_dict[k]
                else:
                    print('Error: invalid CPU brand (' + str(constraint_dict[k]) + ')')
            elif k=='gpu_brand':
                if constraint_dict[k] in ['NVIDIA', 'PreferNVIDIA', 'PreferRadeon', 'Radeon']:
                    self.gpu_brand = constraint_dict[k]
                else:
                    print('Error: invalid GPU brand (' + str(constraint_dict[k]) + ')')
            elif k=='min_ram':
                if constraint_dict[k] in ['16', '32', '64', '128']:
                    self.min_ram = int(constraint_dict[k])
                else:
                    print('Error: invalid Minimum RAM (' + str(constraint_dict[k]) + ')')
            elif k=='max_budget':
                budget=int(constraint_dict[k])
                if 0 <= budget <= 10000:
                    self.max_budget = budget
                else:
                    print('Error: invalid Maximum Budget (' + str(constraint_dict[k]) + ')')
            else:
                print('Error: did not understand key (' + k + ')')

    def ok(self, configuration, noncompliant=None):
        # Check the configuration passed in and return True if all constraints are met
        # If noncompliant is not None (should be an empty list), pass back which constraints are not met
        return True

class TestConstraints(unittest.TestCase):
    def test_constraints_good(self):
        constraints1 = { 'cpu_brand' : 'Intel', 'gpu_brand' : 'PreferNVIDIA',
                         'min_ram' : '16', 'max_budget' : '1500' }
        uut=Constraints(constraints1)
        self.assertEqual(uut.cpu_brand, 'Intel')
        self.assertEqual(uut.gpu_brand, 'PreferNVIDIA')
        self.assertEqual(uut.min_ram, 16)
        self.assertEqual(uut.max_budget, 1500)

    def test_constraints_bad(self):
        print('\nExpect to see invalid results in this test.')
        constraints2 = { 'cpu_brand' : 'INTEL', 'gpu_brand' : 'RadeonXYZ',
                         'min_ram' : '7', 'max_budget' : '-5' }
        uut=Constraints(constraints2)
        self.assertEqual(uut.cpu_brand, None)
        self.assertEqual(uut.gpu_brand, None)
        self.assertEqual(uut.min_ram, None)
        self.assertEqual(uut.min_ram, None)

if __name__ == "__main__":
    unittest.main()
