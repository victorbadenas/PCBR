# System/standard imports
import numpy

# Our imports

# Constants

# Module-global data

# Function definitions

# Class definitions
class AdaptPC:
    """class used to perform the Reuse function and adapt a case to the input request
    """
    def __init__(self, cpu_table, gpu_table):
        """initialize the Reuse class and load domain knowledge/rules/etc.
        """
        self.cpu_table = cpu_table
        self.gpu_table = gpu_table

    def adapt(self, fromCaseBase, constraints=None):
        """start with case from case base and then apply domain knowlege to adapt it to user's needs
        """
        adaptedSolution = fromCaseBase.copy()

        # Use domain knowledge to adapt it

        # TODO: May need to make this a loop and break when "good enough"
        good_enough = False
        while not good_enough:
            # First check if there are any unmet constraints
            self.check_constraints(adaptedSolution,constraints)

            # Next check if there are any HW incompatibilities
            self.check_compatibility(adaptedSolution)

            # Check if there are any optimizations
            self.check_optimizations(adaptedSolution)

            good_enough = True

        # May need to convert from benchmark to CPU/GPU here? Adapted solution probably needs a bit more
        # than just the numeric data

        return adaptedSolution

    def check_constraints(self, solution, constraints):
        print('checking constraints...')
        return

    def check_compatibility(self, solution):
        print('checking compatibility...')
        # if CPU==AMD GPU can't be Integrated
        # if task==(ML|Gaming) require GPU
        return

    def check_optimizations(self, solution):
        print('checking for additional optimizations...')
        return