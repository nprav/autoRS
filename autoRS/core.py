"""Core Table classes to represent Data."""
# %% Import Necessary Modules

# Standard library imports
from collections.abc import MutableMapping
from typing import Tuple, Union, Dict, Optional, Iterator, Sequence
from copy import deepcopy

# Third party imports
import numpy as np

# Local application imports
from autoRS.typing import array_like_1d, array_like_2d


# %% Class definition


class Table(MutableMapping):
    """Class that represents 2D numerical data with an optional specified index column.
    """

    _invalid_column_names = {"Index"}

    def __init__(
        self,
        raw_data: Union[Dict[str, array_like_1d], array_like_1d, array_like_2d] = None,
        index: Optional[array_like_1d] = None,
        column_names: Optional[Sequence[str]] = None,
        index_name: str = "Index",
    ):

        # Setup main data private variables
        self._data: Optional[np.ndarray] = None
        self._index: Optional[np.ndarray] = None
        self._column_dict: Dict[str, np.ndarray] = {}

        # Setup index name and validate column_names
        self.index_name = index_name
        self._invalid_column_names.add(index_name)
        if column_names is not None and set(column_names).intersection(
            self._invalid_column_names
        ):
            raise ValueError("Invalid column name(s).")

        # Setup main columns (automatically sets up index as well)
        if raw_data is not None:
            if isinstance(raw_data, dict):
                self.data, self.column_names = self._parse_data_dict(raw_data)
            else:
                self.data = self._parse_data_not_dict(raw_data)

        # Setup the index (only required if an index is provided but data is not)
        if index is not None:
            self.index = index

        if column_names is not None:
            self.column_names = column_names

    # Properties
    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            nan_data = np.empty(shape=self.shape)
            nan_data[:] = np.nan
            return nan_data
        else:
            return self._data

    @data.setter
    def data(self, raw_data: Union[array_like_1d, array_like_2d]) -> None:
        data = self._parse_data_not_dict(raw_data)
        if self.data.size != 0 and self.data.shape != data.shape:
            raise ValueError("Shape mis-match.")
        if self.index is not None and len(self.index) != data.shape[0]:
            raise ValueError("Data does not match length of existing index.")
        else:
            self._data = data
        if self.index is None:
            self._set_index_from_data()
        if not self.column_names:
            self._set_column_names_from_data()

    @property
    def index(self) -> np.ndarray:
        return self._index

    @index.setter
    def index(self, index: array_like_1d) -> None:
        index = np.asarray(index)
        if self.index is not None and len(index) != len(self.index):
            raise ValueError("Invalid size. New index does not match data length.")
        else:
            self._index = index

    @property
    def column_names(self) -> Tuple[str]:
        return tuple(self._column_dict.keys())

    @column_names.setter
    def column_names(self, column_names: Optional[Sequence[str]] = None) -> None:
        if self.data.size != 0 and self.data.shape[1] != len(column_names):
            raise ValueError(
                "Size Mis-match between provided sequence of column "
                "names and defined data."
            )
        else:
            self._column_dict = {str(key): i for i, key in enumerate(column_names)}

    @property
    def shape(self) -> Tuple[int, int]:
        x, y = 0, 0
        if self.index is not None:
            x = len(self.index)
        y = len(self.column_names)
        return x, y

    # Magic methods
    def __delitem__(self, key: str) -> None:
        pass

    def __getitem__(self, key: str) -> array_like_1d:
        if key == self.index_name:
            return self.index
        idx = self._column_dict[key]
        return self.data[:, idx]

    def __setitem__(self, key: str, value: array_like_1d) -> None:
        pass

    def __len__(self) -> int:
        return self.shape[1]

    def __iter__(self) -> Iterator[Tuple[str, array_like_1d]]:
        pass

    # Helper methods
    @staticmethod
    def _parse_data_dict(
        raw_data: Dict[str, array_like_1d]
    ) -> [np.ndarray, Dict[str, int]]:
        if len(raw_data) == 0:
            raise ValueError("Input Dictionary is empty.")
        elif len(set([len(value) for value in raw_data.values()])) > 1:
            raise ValueError("Invalid input dictionary. Mis-matched column lengths.")

        data = np.asarray(deepcopy(list(raw_data.values())), dtype=float).T

        # If the input is 1d, convert it to 2d
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))

        column_names = list(raw_data.keys())

        return data, column_names

    @staticmethod
    def _parse_data_not_dict(
        raw_data: Union[array_like_1d, array_like_2d],
    ) -> [np.ndarray, Dict[str, int]]:

        data = np.asarray(deepcopy(raw_data), dtype=float)

        # If the input is 1d, convert it to 2d
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))

        if any([x == 0 for x in data.shape]):
            raise ValueError("Input contains empty array.")

        return data

    def _set_index_from_data(self):
        self.index = np.arange(0, self.data.shape[0])

    def _set_column_names_from_data(self):
        self.column_names = [f"Col{i}" for i in range(self.data.shape[1])]
