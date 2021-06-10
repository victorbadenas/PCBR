# System/standard imports
import logging
import numpy as np

# Our imports
from user_request import UserRequest

# Constants

# Module-global data
reuse_logger = logging.getLogger('reuse')

source_columns = ['CPU Mark', 'Capacity', 'Capacity', 'Capacity', 'Benchmark', 'Boolean State']
target_columns = ['CPU Name', 'Capacity', 'Capacity', 'Capacity', 'GPU Name', 'Boolean State']
price_columns = ['MSRP', 'Price', 'Price', 'Price', 'MSRP', 'Price']


# Function definitions
def map_to_closest(adapted_solution, mappers, scalers):
        # Mapping to closest real component.
        # Putting into a function since it's a human-readable way to monitor
        # transformations as they are applied. This function is also important
        # because it calculates the price of the assembled solution.

        # Copy data so we don't destroy it
        tmp_adapted_solution = adapted_solution.copy()

        # TODO: Should we start the price off higher than 0 to account for miscellaneous things
        #       like motherboard, case, etc., and be a fairer comparison to the cases? I'm
        #       thinking maybe 150-200€ might be a fair starting point...
        solution_price = 0
        for idx in range(len(tmp_adapted_solution) - 1):
            tmp_adapted_solution[idx] = mappers[idx].transform(np.array(tmp_adapted_solution[idx]),
                                                        from_col=mappers[idx].scaler_columns[0],
                                                        to_col=target_columns[idx])[0]
            solution_price += mappers[idx].transform(np.array(tmp_adapted_solution[idx]),
                                                     from_col=target_columns[idx],
                                                     to_col=price_columns[idx],
            )[0]
        tmp_adapted_solution[-1] = np.round(solution_price, 2)

        # Transformation of Log2 components.
        for idx in range(1, 4):
            tmp_adapted_solution[idx] = np.round(
                np.power(
                    2,
                    scalers[idx-1].inverse_transform(
                        [[tmp_adapted_solution[idx]]])[0][0]
                ) - 1
            )
        return tmp_adapted_solution

def map_to_numeric(symbolic, mappers, scalers, additional_info=None):
        # Copy data so we don't destroy it
        numeric = symbolic.copy()

        # If additional info is requested, set up the structure
        if additional_info == []:
            cpu_brand=mappers[0].transform(np.array(numeric[0]),
                                            from_col=target_columns[0],
                                            to_col='Manufacturer')[0]
            gpu_brand=mappers[4].transform(np.array(numeric[4]),
                                            from_col=target_columns[4],
                                            to_col='Manufacturer')[0]
            additional_info.append(cpu_brand)
            additional_info.append(gpu_brand)

        # Convert symbolic things (CPU/GPU names) to numbers
        numeric[0]=mappers[0].transform(np.array(numeric[0]),
                                        from_col=target_columns[0],
                                        to_col=mappers[0].scaler_columns[0])[0]
        numeric[4]=mappers[4].transform(np.array(numeric[4]),
                                        from_col=target_columns[4],
                                        to_col=mappers[4].scaler_columns[0])[0]

        # Transformation of Log2 components.
        numeric[1:4] = np.log2(numeric[1:4])

        for i in range(1,4):
            numeric[i]=mappers[i].scaler['scaler'].transform(np.array(numeric[i]).reshape(-1,1))
            numeric[i]=mappers[i].transform(numeric[i],
                                            from_col=target_columns[i],
                                            to_col=mappers[i].scaler_columns[0])[0]

        # Note: No transformations required for optical drive or price

        return numeric

# Class definitions
class AdaptPC:
    """
    Class used to perform the Reuse function and adapt a case to the input request
    """

    def __init__(self, pcbr):
        """initialize the Reuse class and load domain knowledge/rules/etc.
        """
        self.pcbr = pcbr

    def adapt(self, nearest_neighbors, distances, mappers, scalers, user_request):
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
        reuse_logger.debug('Numeric representation: ' + str(adapted_solution))
        closest=map_to_closest(adapted_solution, mappers, scalers)
        reuse_logger.debug('Configuration after weighted adaptation: ' + str(map_to_closest(adapted_solution, mappers, scalers)))
        additional_info=[]
        numeric=map_to_numeric(closest, mappers, scalers,additional_info=additional_info)
        reuse_logger.debug('Numeric representation (closest): ' + str(numeric))
        reuse_logger.debug('Additional info: ' + str(additional_info))

        reuse_logger.debug('Checking constraints and optimizing...')
        reuse_logger.debug('CPU Brand: '  + str(user_request.constraints.cpu_brand))
        reuse_logger.debug('GPU Brand: '  + str(user_request.constraints.gpu_brand))
        reuse_logger.debug('Min RAM: '    + str(user_request.constraints.min_ram))
        reuse_logger.debug('Max budget: ' + str(user_request.constraints.max_budget))
        reuse_logger.debug('Preferences: ' + str(user_request.preferences))
        reuse_logger.debug('Raw Preferences: ' + str(user_request.raw_preferences))

        constraints_ok=user_request.constraints.ok(adapted_solution)
        reuse_logger.debug('Constraints OK: ' + str(constraints_ok))
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

        reuse_logger.debug('Done checking constraints and optimizing.')

        # Convert from numeric to human-readable, mapping to closest values where there isn't an
        # exact match
        adapted_solution=map_to_closest(adapted_solution, mappers, scalers)

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
