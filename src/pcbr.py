import os, sys
sys.path.append(os.path.dirname(__file__))

from data.preprocessor import read_initial_cbl, read_cpu_table, read_gpu_table
from neighbors.knn import KNeighborsClassifier

class PCBR:
    def __init__(self, cbl_path='../data/pc_specs_v2.csv',
                       cpu_path='../data/cpu_table.csv',
                       gpu_path='../data/gpu_table.csv'):
        case_library = read_initial_cbl(path=cbl_path, cpu_path=cpu_path, gpu_path=gpu_path)
        self.target_attributes = case_library[:, :7]
        self.source_attributes = case_library[:, 7:]
        self.cpu_table = read_cpu_table(path=cpu_path)
        self.gpu_table = read_gpu_table(path=gpu_path)

    def retrieve(self, newInstance=None, n_neighbors=2):
        if newInstance is None:
            newInstance = self.source_attributes[2].reshape(1, -1)
        clf = KNeighborsClassifier(n_neighbors=2).fit(self.source_attributes, self.target_attributes)
        return clf.predict(newInstance)


if __name__ == '__main__':
    pcbr = PCBR()
    ret = pcbr.retrieve()
    print(ret, ret.shape)
