import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path

_NUMERIC_KINDS = set('buifc')


def is_numeric(array):
    """Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    Parameters
    ----------
    array : array-like
        The array to check.

    Returns
    -------
    is_numeric : `bool`
        True if the array has a numeric datatype, False if not.

    """
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS


class Mapper:
    def __init__(self, dataframe: pd.DataFrame, scaler_columns=None, scaler=None):
        self.data = dataframe
        self.scaler = scaler
        self.scaler_columns = scaler_columns
        if scaler is not None and scaler_columns is not None:
            self._scale_data()

    @classmethod
    def from_csv(cls, path: Union[str, Path] = None, **kwargs):
        if path is None:
            raise ValueError('path is required for Mapper.from_csv(path) method')
        path = Path(path)
        if not path.exists():
            raise OSError(f'file {path} does not exist')
        return cls(pd.read_csv(path), **kwargs)

    @property
    def columns(self):
        return self.data.columns

    @property
    def shape(self):
        return self.data.shape

    def transform(self, X, from_col, to_col):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a np.array')
        if from_col not in self.data.columns or to_col not in self.data.columns:
            raise ValueError(f'Columns {(from_col, to_col)} not in dataframe')

        source_data = self.data[from_col].to_numpy()
        target_data = self.data[to_col].to_numpy()
        if is_numeric(source_data):
            nn_index = np.argmin(np.abs(np.subtract(source_data, X)))
            return target_data[nn_index]
        else:
            map_dict = dict(zip(source_data, target_data))
            vfunc = np.vectorize(lambda x: map_dict[x])
            return vfunc(X)

    def _scale_data(self):
        for column in self.scaler_columns:
            if self.scaler['log2']:
                self.data[column] = np.log2(
                    self.data[column],
                    where=self.data[column] != 0
                )
            if hasattr(self.scaler['scaler'], 'transform'):
                self.data[column] = self.scaler['scaler'].transform(
                    self.data[column].to_numpy().reshape(-1, 1)
                )
