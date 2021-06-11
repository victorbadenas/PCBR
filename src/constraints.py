import unittest


class Constraints:
    def __init__(self, constraint_dict={}):
        """initialize the constraints structure using a dictionary of keys/values from user input
        """

        # Initialize everything to default (no constraints)
        self.cpu_brand = None
        self.gpu_brand = None
        self.min_ram = None
        self.max_budget = None
        self.optical_drive = None

        # Potential for Future: If we want to specify specific CPU/GPU
        # self.cpu=None
        # self.gpu=None

        for k in constraint_dict:
            if k == 'cpu_brand':
                if constraint_dict[k] in ['Intel', 'PreferIntel', 'PreferAMD', 'AMD']:
                    self.cpu_brand = constraint_dict[k]
                else:
                    print('Error: invalid CPU brand (' + str(constraint_dict[k]) + ')')
            elif k == 'gpu_brand':
                if constraint_dict[k] in ['NVIDIA', 'PreferNVIDIA', 'PreferAMD', 'AMD']:
                    self.gpu_brand = constraint_dict[k]
                else:
                    print('Error: invalid GPU brand (' + str(constraint_dict[k]) + ')')
            elif k == 'min_ram':
                if constraint_dict[k] in ['16', '32', '64', '128']:
                    self.min_ram = int(constraint_dict[k])
                else:
                    print('Error: invalid Minimum RAM (' + str(constraint_dict[k]) + ')')
            elif k == 'max_budget':
                budget = int(constraint_dict[k])
                if 0 <= budget <= 10000:
                    self.max_budget = budget
                else:
                    print('Error: invalid Maximum Budget (' + str(constraint_dict[k]) + ')')
            elif k == 'optical_drive':
                if constraint_dict[k] in ['yes', 'no']:
                    self.optical_drive = constraint_dict[k]
                else:
                    print('Error: invalid Optical Drive option(' + str(constraint_dict[k]) + ')')
            else:
                print('Error: did not understand key (' + k + ')')

    def ok(self, configuration, selected_cpu_brand, selected_gpu_brand):
        # Check the configuration passed in and return a list of booleans for whether each constraint
        # is met or not. Order: CPU Brand, GPU Brand, RAM, Budget, Optical Drive
        # CPU/GPU brands are passed in along with the configuration since this class doesn't have access
        # to the appropriate lookup tables

        cpu_ok = True
        gpu_ok = True
        ram_ok = True
        budget_ok = True
        optical_ok = True

        if self.cpu_brand is not None:
            # Only consider unmet if it is as "MUST" (not a prefer) and doesn't match
            if self.cpu_brand == 'Intel' and selected_cpu_brand != 'Intel':
                cpu_ok = False
            if self.cpu_brand == 'AMD' and selected_cpu_brand != 'AMD':
                cpu_ok = False

        if self.gpu_brand is not None:
            # Only consider unmet if it is as "MUST" (not a prefer) and doesn't match
            if self.gpu_brand == 'NVIDIA' and selected_gpu_brand != 'NVIDIA':
                gpu_ok = False
            if self.gpu_brand == 'AMD' and selected_gpu_brand != 'AMD':
                gpu_ok = False

        if self.min_ram is not None:
            if configuration[1] < self.min_ram:
                ram_ok = False

        if self.max_budget is not None:
            if configuration[6] > self.max_budget:
                budget_ok = False

        if self.optical_drive is not None:
            if self.optical_drive == 'yes' and configuration[5] == 0:
                optical_ok = False
            elif self.optical_drive == 'no' and configuration[5] == 1:
                optical_ok = False

        return [ cpu_ok, gpu_ok, ram_ok, budget_ok, optical_ok ]


class TestConstraints(unittest.TestCase):
    def test_constraints_good(self):
        constraints1 = {'cpu_brand': 'Intel', 'gpu_brand': 'PreferNVIDIA',
                        'min_ram': '16', 'max_budget': '1500', 'optical_drive': 'yes'}
        uut = Constraints(constraints1)
        self.assertEqual(uut.cpu_brand, 'Intel')
        self.assertEqual(uut.gpu_brand, 'PreferNVIDIA')
        self.assertEqual(uut.min_ram, 16)
        self.assertEqual(uut.max_budget, 1500)
        self.assertEqual(uut.optical_drive, 'yes')

    def test_constraints_bad(self):
        print('\nExpect to see invalid results in this test.')
        constraints2 = {'cpu_brand': 'INTEL', 'gpu_brand': 'RadeonXYZ',
                        'min_ram': '7', 'max_budget': '-5',
                        'optical_drive': 'uhhh... what?'}
        uut = Constraints(constraints2)
        self.assertEqual(uut.cpu_brand, None)
        self.assertEqual(uut.gpu_brand, None)
        self.assertEqual(uut.min_ram, None)
        self.assertEqual(uut.max_budget, None)
        self.assertEqual(uut.optical_drive, None)


if __name__ == "__main__":
    unittest.main()
