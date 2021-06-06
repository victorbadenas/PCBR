import os, sys
sys.path.append(os.path.dirname(__file__))
from collections import namedtuple

from data.preprocessor import read_initial_cbl, read_cpu_table, read_gpu_table
from neighbors.knn import KNeighborsClassifier
from adapt_pc import AdaptPC
from constraints import Constraints

UserRequest = namedtuple('UserRequest', ['instance', 'constraints'])

class PCBR:
    def __init__(self, cbl_path='../data/pc_specs_v2.csv',
                       cpu_path='../data/cpu_table.csv',
                       gpu_path='../data/gpu_table.csv'):
        case_library = read_initial_cbl(path=cbl_path, cpu_path=cpu_path, gpu_path=gpu_path)
        # Split into "source" (preferences) and "target" (PC specs)
        self.target_attributes = case_library[:, :7]
        self.source_attributes = case_library[:, 7:]
        self.cpu_table = read_cpu_table(path=cpu_path)
        self.gpu_table = read_gpu_table(path=gpu_path)
        print('case library: ' + str(case_library.shape))
        print('cpu table: ' + str(self.cpu_table.shape))
        print('gpu table: ' + str(self.gpu_table.shape))
        self.adapt_pc = AdaptPC(self.cpu_table, self.gpu_table)

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
            newInstance = self.source_attributes[2].reshape(1, -1)
        print('looking for: ' + str(newInstance))
        clf = KNeighborsClassifier(n_neighbors=2).fit(self.source_attributes, self.target_attributes)
        return clf.predict(newInstance)

    def reuse(self, newInstance=None, constraints=None):
        assert(newInstance is not None)
        print('starting with: ' + str(newInstance))
        adaptedCase = self.adapt_pc.adapt(newInstance,constraints)
        print('adapted to: ' + str(adaptedCase))
        return adaptedCase

if __name__ == '__main__':
    pcbr = PCBR()
    userRequest = pcbr.get_user_request()
    nearestCases = pcbr.retrieve(newInstance=userRequest.instance)
    print(nearestCases, nearestCases.shape)
    proposedSolution = pcbr.reuse(nearestCases[0,0],userRequest.constraints) # Just pass one for now, but may want the 2nd-nearest neighbor too
    # Uncomment as these functions get implemented
    #revisionResult=pcbr.revise(proposedSolution)
    #pcbr.retain(proposedSolution, revisionResult)

    # TODO: Should we write new case base to file and exit or just keep looping?
