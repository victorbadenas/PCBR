from pathlib import Path
from typing import Union

def read_file_iterator(file_path: Union[str, Path], sep=','):
    with open(Path(file_path), 'r') as f:
        for line in f:
            yield line.strip().split(sep)

def read_file(file_path: Union[str, Path], sep=','):
    with open(Path(file_path), 'r') as f:
        return list(map(lambda line: line.strip().split(sep), f))
