import os, sys
sys.path.append(os.path.dirname(__file__))
from collections import namedtuple

import logging

from data.preprocessor import read_initial_cbl, read_table
from data.mapper import Mapper
from neighbors.knn import KNeighborsClassifier
from adapt_pc import AdaptPC
from constraints import Constraints

# Logger objects
pcbr_logger=logging.getLogger('pcbr')
retrieve_logger=logging.getLogger('retrieve')
reuse_logger=logging.getLogger('reuse')
revise_logger=logging.getLogger('revise')
retain_logger=logging.getLogger('retain')

UserRequest = namedtuple('UserRequest', ['instance', 'constraints'])

def setup_logging():
    # Set up logging subsystem. Level should be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    logging.basicConfig()

    # Choose the modules whose output you want to see here. INFO is a good default level, but DEBUG
    # may be useful during development of your module
    pcbr_logger.setLevel(logging.INFO)
    retrieve_logger.setLevel(logging.INFO)
    reuse_logger.setLevel(logging.INFO)
    revise_logger.setLevel(logging.INFO)
    retain_logger.setLevel(logging.INFO)

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

        pcbr_logger.debug('case library: ' + str(case_library.shape))
        pcbr_logger.debug('cpu table: ' + str(self.cpu_table.shape))
        pcbr_logger.debug('gpu table: ' + str(self.gpu_table.shape))
        pcbr_logger.debug('ram table: ' + str(self.ram_table.shape))
        pcbr_logger.debug('ssd_table: ' + str(self.ssd_table.shape))
        pcbr_logger.debug('hdd_table: ' + str(self.hdd_table.shape))
        pcbr_logger.debug('opt_drive_table: ' + str(self.opt_drive_table.shape))

        # initialize the adapt_pc object
        self.adapt_pc = AdaptPC(self)
        pcbr_logger.info('Initialization complete!')

    def get_user_request(self):
        # Request input here and return it.
        # For now, appears "None" is handled well by retrieve step and it defaults to a case in the library
        # Either need to pre-process the request here or in the retrieve step.
        # Also need to pass along some extra metadata, such as constraints.
        constraints=Constraints()
        userReqRv=UserRequest(None,constraints)
        return userReqRv

    def retrieve(self, newInstance=None, n_neighbors=2):
        if newInstance is None:
            newInstance = self.source_attributes.iloc[2].to_numpy().reshape(1, -1)
        pcbr_logger.debug('looking for: ' + str(newInstance))
        clf = KNeighborsClassifier(n_neighbors=2).fit(self.source_attributes.to_numpy(), self.target_attributes.to_numpy())
        return clf.predict(newInstance)

    def reuse(self, newInstance=None, constraints=None):
        assert(newInstance is not None)
        pcbr_logger.debug('starting with: ' + str(newInstance))
        adaptedCase = self.adapt_pc.adapt(newInstance,constraints)
        pcbr_logger.debug('adapted to: ' + str(adaptedCase))
        return adaptedCase

if __name__ == '__main__':
    setup_logging()

    pcbr = PCBR()
    userRequest = pcbr.get_user_request()
    nearestCases = pcbr.retrieve(newInstance=userRequest.instance)
    pcbr_logger.debug(nearestCases)
    pcbr_logger.debug(nearestCases.shape)
    proposedSolution = pcbr.reuse(nearestCases[0,0],userRequest.constraints) # Just pass one for now, but may want the 2nd-nearest neighbor too
    # Uncomment as these functions get implemented
    #revisionResult=pcbr.revise(proposedSolution)
    #pcbr.retain(proposedSolution, revisionResult)

    # TODO: Should we write new case base to file and exit or just keep looping?
