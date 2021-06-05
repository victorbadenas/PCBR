from data.preprocessor import read_initial_cbl

if __name__ == '__main__':
    initial_cbl = read_initial_cbl()
    print(initial_cbl[:5])
