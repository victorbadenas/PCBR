import numpy as np
import pandas as pd
from collections import Iterable
import time
import sys


def bytesToString(bytesObject):
    if hasattr(bytesObject, 'decode'):
        return bytesObject.decode()
    return bytesObject


def timer(print_=False):
    def inner2(func):
        def inner(*args, **kwargs):
            st = time.time()
            ret = func(*args, **kwargs)
            if print_:
                print(f"{func.__name__} ran in {time.time()-st:.6f}s")
                return ret
            else:
                delta = time.time() - st
                return ret, delta
        return inner
    return inner2


def separateOutput(msg):
    def inner2(f):
        def inner(*args, **kwargs):
            print(f"---------------------{msg}---------------------")
            return f(*args, **kwargs)
        return inner
    return inner2


def ndcorrelate(X, Y):
    X = pd.DataFrame(X)
    X["y"] = pd.Series(Y)
    corr = X.corr().to_numpy()[-1, :-1]
    return corr


def getSizeOfObject(obj):
    size = 0.0
    if hasattr(obj, '__dict__'):  # explore all attributes in a class
        for _, v in obj.__dict__.items():
            size += getSizeOfObject(v)
    elif isinstance(obj, (int, float, str, np.ndarray)):
        size += sys.getsizeof(obj)
    elif isinstance(obj, dict):
        for k in obj:
            size += getSizeOfObject(obj[k])
    elif isinstance(obj, Iterable):
        for item in obj:
            size += getSizeOfObject(item)
    return size
