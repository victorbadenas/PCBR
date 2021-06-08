# System/standard imports
import logging
import numpy

# Our imports

# Constants

# Module-global data
reuse_logger=logging.getLogger('reuse')

# Function definitions

# Class definitions
class AdaptPC:
    """class used to perform the Reuse function and adapt a case to the input request
    """
    def __init__(self, pcbr):
        """initialize the Reuse class and load domain knowledge/rules/etc.
        """
        self.pcbr=pcbr

    def adapt(self, nearest_neighbors, user_request):
        """start with case from case base and then apply domain knowlege to adapt it to user's needs
        """
        adapted_solution = nearest_neighbors[0,0].copy()

        # Use domain knowledge to adapt it

        # TODO: May need to make this a loop and break when "good enough"
        good_enough = False
        while not good_enough:
            # First check if there are any unmet constraints
            self.check_constraints(adapted_solution,user_request.constraints)

            # Next check if there are any HW incompatibilities
            self.check_compatibility(adapted_solution)

            # Check if there are any optimizations
            self.check_optimizations(adapted_solution)

            good_enough = True

        # May need to convert from benchmark to CPU/GPU here? Adapted solution probably needs a bit more
        # than just the numeric data

        return adapted_solution

    def check_constraints(self, solution, constraints):
        reuse_logger.debug('checking constraints...')
        reuse_logger.debug(constraints)
        reuse_logger.debug('cpu brand: ' + str(constraints.cpu_brand))
        return

    def check_compatibility(self, solution):
        reuse_logger.debug('checking compatibility...')
        # if CPU==AMD GPU can't be Integrated
        # if task==(ML|Gaming) require GPU
        return

    def check_optimizations(self, solution):
        reuse_logger.debug('checking for additional optimizations...')
        return
