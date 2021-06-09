# System/standard imports
import logging
import numpy as np

# Our imports

# Constants

# Module-global data
reuse_logger = logging.getLogger('reuse')


# Function definitions

# Class definitions
class AdaptPC:
    """
    Class used to perform the Reuse function and adapt a case to the input request
    """

    def __init__(self, pcbr):
        """initialize the Reuse class and load domain knowledge/rules/etc.
        """
        self.pcbr = pcbr

    def adapt(self, nearest_neighbors, distances, mappers, scalers):
        """start with case from case base and then apply domain knowledge to adapt it to user's needs
        """

        # Weighted average given distances to nearest neighbors so as to create new PC.
        sims = [1/(distance + 0.1) for distance in distances]
        norm_sims = np.array([sim/np.sum(sims) for sim in sims])[0]

        adapted_solution = [np.sum([norm_sims[neighbor_idx]*nearest_neighbors[neighbor_idx][target_col]
                                    for neighbor_idx in range(nearest_neighbors.shape[0])])
                            for target_col in range(nearest_neighbors.shape[1])]

        # Use domain knowledge to adapt it
        # Kevin: Constraints will be solved after the adaptation stage. It takes into account possible compatibility
        #        issues. The solution is already optimized based on the weighted kNN.
        """
        # TODO: May need to make this a loop and break when "good enough"
        good_enough = False
        while not good_enough:
            # First check if there are any unmet constraints
            self.check_constraints(adapted_solution, user_request.constraints)

            # Next check if there are any HW incompatibilities
            self.check_compatibility(adapted_solution)

            # Check if there are any optimizations
            self.check_optimizations(adapted_solution)

            good_enough = True
        """

        # May need to convert from benchmark to CPU/GPU here? Adapted solution probably needs a bit more
        # than just the numeric data.
        # Kevin: Done!!

        # Mapping to closest real component.
        target_columns = ['CPU Name', 'Capacity', 'Capacity', 'Capacity', 'GPU Name', 'Boolean State']
        for idx in range(len(adapted_solution) - 1):
            adapted_solution[idx] = mappers[idx].transform(np.array(adapted_solution[idx]),
                                                           from_col=mappers[idx].scaler_columns[0],
                                                           to_col=target_columns[idx])

        # Transformation of Log2 components.
        for idx in range(1, 4):
            adapted_solution[idx] = np.round(
                np.power(
                    2,
                    scalers[idx-1].inverse_transform(
                        [[adapted_solution[idx]]])[0][0]
                ) - 1
            )
        # TODO: Transform price using sum of price components instead of inverse_transform of weighted average.
        adapted_solution[-1] = scalers[-1].inverse_transform([[adapted_solution[-1]]])[0][0]

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
