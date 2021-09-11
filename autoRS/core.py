"""Core Table classes to represent Data."""
# %% Import Necessary Modules

# Standard library imports
from collections.abc import MutableMapping
from collections import OrderedDict
from typing import Tuple, Union, Dict, Optional, Iterator, Sequence, List
from copy import deepcopy

# Third party imports
import numpy as np

# Local application imports
from autoRS.typing import numeric, array_like_1d, array_like_2d


# %% Class definition


class Table(MutableMapping):
    """Class that represents 2D numerical data with an optional specified index column.
    """

    _invalid_column_names = {"Index"}

    def __init__(
        self,
        data: Union[Dict[str, array_like_1d], array_like_1d, array_like_2d] = None,
        index: Optional[array_like_1d] = None,
        column_names: Optional[Sequence[str]] = None,
        index_name: str = "Index",
    ):

        # Setup index name and validate column_names
        self.index_name = index_name
        self._invalid_column_names.add(index_name)
        if column_names is not None and set(column_names).intersection(
            self._invalid_column_names
        ):
            raise ValueError("Invalid column name(s).")

        # Setup main columns
        if isinstance(data, dict):
            self._data, self._column_dict = self._parse_data_dict(data)
        else:
            self._data, self._column_dict = self._parse_data_not_dict(
                data, column_names
            )

        # Setup the index
        if index is None:
            self.index = np.arange(0, self.shape[0])
        else:
            if self.data is not None and len(index) != self.shape[0]:
                raise ValueError("Invalid index. Length does not match data length.")
            self.index = np.asarray(index)

    # Properties
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        # TODO: Raise error if sizes do not match
        if self.data is not None:
            if any(self.shape != data):
                raise ValueError("Shape mis-match.")
        else:
            self._data = np.array(deepcopy(data))

    @property
    def columns(self):
        # TODO: add columns setter
        return tuple(self._column_dict.keys())

    @property
    def shape(self):
        if self.data is not None:
            return self.data.shape
        elif self.index is not None:
            return self.index.shape
        else:
            return np.array([]).shape

    # Magic methods
    def __delitem__(self, v: str) -> None:
        pass

    def __getitem__(self, key: str) -> array_like_1d:
        if key == self.index_name:
            return self.index
        idx = self._column_dict[key]
        return self.data[:, idx]

    def __setitem__(self, k: str, v: array_like_1d) -> None:
        pass

    def __len__(self) -> int:
        return self.shape[1]

    def __iter__(self) -> Iterator[Tuple[str, array_like_1d]]:
        pass

    # Helper methods
    def _parse_data_dict(
        self, data: Dict[str, array_like_1d]
    ) -> [np.ndarray, Dict[str, int]]:
        if len(data) == 0:
            raise ValueError("Input Dictionary is empty.")
        elif len(set([len(value) for value in data.values()])) > 1:
            raise ValueError("Invalid input dictionary. Mis-matched column lengths.")

        _data = np.asarray(deepcopy(list(data.values())), dtype=float).T
        _column_dict = {key: i for i, key in enumerate(data.keys())}
        return _data, _column_dict

    def _parse_data_not_dict(
        self,
        data: Union[array_like_1d, array_like_2d],
        column_names: Optional[Sequence[str]],
    ) -> [np.ndarray, Dict[str, int]]:

        if len(data) == 0 or not all(data):
            raise ValueError("Input contains empty array.")
        elif len(set(type(value) for value in data)) > 1:
            raise ValueError("Invalid input. Mis-matched types.")
        elif (
            type(data[0]) not in (int, float)
            and len(set([len(value) for value in data])) > 1
        ):
            raise ValueError("Invalid input. Mis-matched column lengths.")

        _data = np.asarray(deepcopy(data), dtype=float)

        # If the input is 1d, convert it to 2d
        if len(_data.shape) == 1:
            _data = _data.reshape((-1, 1))
        if column_names is None:
            _column_dict = {i: i for i in range(_data.shape[1])}
        else:
            _column_dict = {key: i for i, key in enumerate(column_names)}
        return _data, _column_dict
