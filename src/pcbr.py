import os, sys
import logging
import numpy as np

sys.path.append(os.path.dirname(__file__))

from data.preprocessor import read_initial_cbl, read_table
from data.mapper import Mapper
from utils.io import read_file
from neighbors.knn import KNeighborsClassifier
from adapt_pc import AdaptPC
from constraints import Constraints

# Logger objects
pcbr_logger=logging.getLogger('pcbr')
retrieve_logger=logging.getLogger('retrieve')
reuse_logger=logging.getLogger('reuse')
revise_logger=logging.getLogger('revise')
retain_logger=logging.getLogger('retain')

def setup_logging():
    # Set up logging subsystem. Level should be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    logging.basicConfig()

    # Choose the modules whose output you want to see here. INFO is a good default level, but DEBUG
    # may be useful during development of your module
    pcbr_logger.setLevel(logging.DEBUG)
    retrieve_logger.setLevel(logging.INFO)
    reuse_logger.setLevel(logging.DEBUG)
    revise_logger.setLevel(logging.INFO)
    retain_logger.setLevel(logging.INFO)

class UserRequest:
    def __init__(self, profile_str, pref_str, constraints_str, scalers):
        self.profile=self._process_profile(profile_str, scalers)
        self.preferences=self._process_preferences(pref_str, scalers)
        self.constraints=self._process_constraints(constraints_str, scalers)

    def _process_profile(self, profile_str:str, scalers:dict=None) -> np.ndarray:
        # Input format: Experience, WFH, Primary Use, Budget, Replace, Office, Photoshop, VideoChat, ML, Compilers, HighPerformanceGames, LowPerformanceGames
        # Input example (exactly matches case 2): '2, 1, Programming, 1, 3, 1, 0, 0, 0, 1, 0, 0'
        # Output format: numpy array suitable to look up nearest case in case library (Note: you will need to
        #                apply pre-processing to obtain the correct output format)
        # Example output: [[0.25       1.         0.6        0.         0.66666667 1.
        #                   0.         0.         0.         1.         0.         0.        ]]

        # TODO: Fill in this function. For now, None ok to return.
        # Also TODO: Convert all the function header comments to nice docstrings or delete.
        return None

    def _process_preferences(self, pref_str:str, scalers:dict=None) -> np.ndarray:
        # Input format: Preferences matrix survey answers (string). Importance on scale of 1-5, where 1 is least
        #               and 5 is most. Categories are: budget, performance, multi-tasking, gaming,
        #               streaming videos, editing videos/photos/music, fast startup/shutdown, video chat
        # Input example: '5, 2, 3, 1, 2, 1, 3, 4'
        # Output format: numpy array of answers
        # Example output: [5 2 3 1 2 1 3 4]
        preferences_arr = list(map(int, pref_str.split(',')))
        # TODO: Do we want this normalized? Probably
        # TODO: Or would we rather convert it to a weight matrix to use directly on the output features?
        #       We're not using this yet, so Kevin/Victor, please chime in with your opinions about how
        #       you think we should format and apply this field. For now, I will just return 1's everywhere.
        return np.array(preferences_arr)

    def _process_constraints(self, constraints_str:str, scalers:dict=None) -> Constraints:
        # Input format: string with multiple comma-separated key: value pairs of constraints. 
        # Input example: 'cpu_brand: Intel, gpu_brand: PreferNVIDIA, max_budget: 1000'
        # Output format: Constraints object

        # TODO: Convert string to constraints dict (or change Constraints object to process the string directly)
        constraints_dict = dict()
        for constraint in constraints_str.split(","):
            k, v = constraint.split(':')
            constraints_dict[k.strip()] = v.strip()
        return Constraints(constraints_dict)


class PCBR:
    def __init__(self, cbl_path='../data/pc_specs.csv',
                       cpu_path='../data/cpu_table.csv',
                       gpu_path='../data/gpu_table.csv',
                       ram_path='../data/ram_table.csv',
                       ssd_path='../data/ssd_table.csv',
                       hdd_path='../data/hdd_table.csv',
                       opt_drive_path='../data/optical_drive_table.csv'):

        pcbr_logger.info('Initializing...')
        # read mappers
        # read case library
        case_library, self.transformations = read_initial_cbl(path=cbl_path, 
            cpu_path=cpu_path,
            gpu_path=gpu_path,
            ram_path=ram_path,
            ssd_path=ssd_path,
            hdd_path=hdd_path,
            opt_drive_path=opt_drive_path
        )

        # Split into "source" (preferences) and "target" (PC specs)
        self.target_attributes = case_library[case_library.columns[:7]]
        self.source_attributes = case_library[case_library.columns[7:]]

        # read component's tables
        self.cpu_mapper = Mapper.from_csv(path=cpu_path, scaler_columns=['CPU Mark'], scaler=self.transformations['CPU'])
        self.gpu_mapper = Mapper.from_csv(path=gpu_path, scaler_columns=['Benchmark'], scaler=self.transformations['GPU'])
        self.ram_mapper = Mapper.from_csv(path=ram_path, scaler_columns=['Capacity'], scaler=self.transformations['RAM (GB)'])
        self.ssd_mapper = Mapper.from_csv(path=ssd_path, scaler_columns=['Capacity'], scaler=self.transformations['SSD (GB)'])
        self.hdd_mapper = Mapper.from_csv(path=hdd_path, scaler_columns=['Capacity'], scaler=self.transformations['HDD (GB)'])
        self.opt_drive_mapper = Mapper.from_csv(path=opt_drive_path, scaler=self.transformations['Optical Drive (1 = DVD, 0 = None)'])

        # initialize the adapt_pc object
        self.adapt_pc = AdaptPC(self)
        pcbr_logger.info('Initialization complete!')

    def get_user_request(self, mock_file=None, mode='one_pass'):
        # Request input here and return it.
        # For now, appears "None" is handled well by retrieve step and it defaults to a case in the library
        # Either need to pre-process the request here or in the retrieve step.
        # Also need to pass along some extra metadata, such as constraints.
        if mock_file is not None:

            # initialize the data and the iteration index pointing to the 
            # instance to be returned in this call
            if not hasattr(self, 'mock_user_requests'):
                self.mock_user_requests = read_file(mock_file, sep='\t')
                self.mock_requests_idx = 0

            # end of loop control.
            if self.mock_requests_idx > len(self.mock_user_requests):
                if mode == 'one_pass':
                    # if we reached the end of the mock list, 
                    # return None as no more instances are available.
                    return None
                elif mode == 'cyclic':
                    # if we reach the end of the mock list and we are in cyclic,
                    # reset the index and start again
                    self.mock_requests_idx = self.mock_requests_idx % len(self.mock_user_requests)

            # load current iteration mock request
            request_strings = self.mock_user_requests[self.mock_requests_idx]
            
            # increment pointer
            self.mock_requests_idx += 1

            # return request built with mock request trings
            return UserRequest(*request_strings, self.transformations)
        else:
            # TODO: CLI request
            user_req_rv = UserRequest(None,None,None)
            return user_req_rv

    def retrieve(self, newInstance=None, n_neighbors=2):
        if newInstance is None:
            newInstance = self.source_attributes.iloc[2].to_numpy().reshape(1, -1)
        pcbr_logger.debug('looking for: ' + str(newInstance))
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(self.source_attributes.to_numpy(), self.target_attributes.to_numpy())
        return clf.predict(newInstance)

    def reuse(self, nearest_cases=None, user_request=None):
        assert(nearest_cases is not None)
        pcbr_logger.debug('starting with: ' + str(nearest_cases))
        adaptedCase = self.adapt_pc.adapt(nearest_cases,user_request)
        pcbr_logger.debug('adapted to: ' + str(adaptedCase))
        return adaptedCase

if __name__ == '__main__':
    setup_logging()

    pcbr = PCBR()

    user_request = pcbr.get_user_request(mock_file='../data/mock_requests.tsv')

    nearest_cases = pcbr.retrieve(newInstance=user_request.profile, n_neighbors=3)
    pcbr_logger.debug(nearest_cases)
    pcbr_logger.debug(nearest_cases.shape)

    proposed_solution = pcbr.reuse(nearest_cases, user_request)

    # Uncomment as these functions get implemented
    #revisionResult=pcbr.revise(proposedSolution)
    #pcbr.retain(proposedSolution, revisionResult)

    # TODO: Should we write new case base to file and exit or just keep looping?
