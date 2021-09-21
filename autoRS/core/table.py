"""Core Table classes to represent Data."""
# %% Import Necessary Modules

# Standard library imports
from collections.abc import MutableMapping
from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Optional, Iterator, Sequence
from copy import deepcopy

# Third party imports
import numpy as np

# Local application imports
from autoRS.typing import array_like_1d, array_like_2d


# %% Class definitions


class Table(ABC, MutableMapping):
    """Abstract class that defines an interface for 2D Tables as an
    extension of dictionary-like objects. The Table o"""

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """Get the numpy ndarray representation of table data."""
        pass

    @property
    @abstractmethod
    def index(self) -> np.ndarray:
        """Get the table index."""
        pass

    @property
    @abstractmethod
    def column_names(self) -> Tuple[str]:
        """Get the names of all columns in the table."""
        pass

    @property
    @abstractmethod
    def index_name(self) -> str:
        """Get the name of the Table index."""
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the data. Similar to the shape of a numpy ndarray."""
        pass


# TODO: Fill out all the docstrings


class FloatTable(Table):
    """Class that represents 2D float-type data with an optional specified index column.
    """

    _invalid_column_names = {"Index"}

    def __init__(
        self,
        raw_data: Union[Dict[str, array_like_1d], array_like_1d, array_like_2d] = None,
        index: Optional[array_like_1d] = None,
        column_names: Optional[Sequence[str]] = None,
        index_name: str = "Index",
    ):
        """Initialize the Table. The Table can be initialized with no data, with only
        data, with only an index, or with data and an index.

        Parameters
        ----------
        raw_data: dict[str: array_like_1d] or array_like_2d or array_like_1d
            Data represented by the table. Can be input as a Dictionary of column name
            keys matched with 1d arrays of data. The arrays must have the same length.
            The data can also be input as a 1d or 2d arrays. However, the column names
            would then have to be defined separately.
        index: array_like_1d or None
            Index for the Table. Defaults to 0, 1, 2, 3 ...
        column_names: Sequence[str] or None
            List of names for the Table columns. Defaults to ["Col0", "Col1", ... etc.].
            The column names will be taken as the keys of `raw_data` if the data has
            been input as a dictionary.
        index_name: str
            Name of the Table index. Defaults to "Index"
        """

        # Setup main data private variables
        self._data: Optional[np.ndarray] = None
        self._index: Optional[np.ndarray] = None
        self._column_dict: Dict[str, int] = {}
        self._index_name: str = index_name

        # Validate column_names
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
        """Get the numpy ndarray representation of table data."""
        if self._data is None:
            nan_data = np.empty(shape=self.shape)
            nan_data[:] = np.nan
            return nan_data
        else:
            return self._data

    @data.setter
    def data(self, raw_data: Union[array_like_1d, array_like_2d]) -> None:
        raw_data = self._parse_data_not_dict(raw_data)
        # If we know the index, compare shapes
        if self.index is not None and self.shape[0] != raw_data.shape[0]:
            raise SizeMismatchError(
                f"Number of rows in data ({raw_data.shape[0]}) does "
                f"not match length of index ({self.shape[0]})."
            )
        # If we know the columns, compare shape
        if self.column_names and self.shape[1] != raw_data.shape[1]:
            raise SizeMismatchError(
                f"Number of columns in data ({raw_data.shape[1]}) does "
                f"not match number of table columns ({self.shape[1]})."
            )

        self._data = raw_data
        if self.index is None:
            self._set_index_from_data()
        if not self.column_names:
            self._set_column_names_from_data()

    @property
    def index(self) -> np.ndarray:
        return self._index

    @index.setter
    def index(self, raw_index: array_like_1d) -> None:
        raw_index = np.asarray(raw_index)
        if self.index is not None and len(raw_index) != len(self.index):
            raise SizeMismatchError(
                f"Table index size is {len(self.index)}. "
                f"Provided index size is {len(raw_index)}."
            )
        else:
            self._index = raw_index

    @property
    def column_names(self) -> Tuple[str]:
        return tuple(self._column_dict.keys())

    @column_names.setter
    def column_names(self, column_names: Optional[Sequence[str]] = None) -> None:
        if self.data.size != 0 and self.data.shape[1] != len(column_names):
            raise SizeMismatchError(
                f"Data contains {self.data.shape[1]} columns. {len(column_names)} "
                f"column names provided."
            )
        else:
            self._column_dict = {str(key): i for i, key in enumerate(column_names)}

    @property
    def index_name(self):
        return self._index_name

    @index_name.setter
    def index_name(self, raw_name: str):
        self._index_name = str(raw_name)

    @property
    def shape(self) -> Tuple[int, int]:
        x, y = 0, 0
        if self.index is not None:
            x = len(self.index)
        y = len(self.column_names)
        return x, y

    # Magic methods/ MutableMapping overrides
    def __delitem__(self, key: str) -> None:
        self._verify_key(key)
        if key == self.index_name:
            raise ValueError("Cannot delete index.")
        else:
            idx = self._column_dict[key]
            self._data = np.delete(self._data, idx, axis=1)
            self._column_dict.pop(key)
            self.column_names = list(self._column_dict.keys())

    def __getitem__(self, key: str) -> array_like_1d:
        self._verify_key(key)
        if key == self.index_name:
            return self.index
        idx = self._column_dict[key]
        return self.data[:, idx]

    def __setitem__(self, key: str, value: array_like_1d) -> None:
        value = np.asarray(value, dtype=float).flatten()

        if key == self.index_name:
            self.index = value
            return

        if self.index is not None and self.shape[0] != len(value):
            raise SizeMismatchError("Input does not match length of Table index.")

        if self.index is None:
            self._set_index_from_data(value)

        new_data = self.data
        if key in self.column_names:
            idx = self._column_dict[key]
            new_data[:, idx] = value
            self.data = new_data
        else:
            if new_data.size == 0:
                new_data = value.reshape(-1, 1)
            else:
                new_data = np.append(new_data, value.reshape(-1, 1), axis=1)
            self._data = new_data
            self.column_names = [*self.column_names, key]

    def __len__(self) -> int:
        return self.shape[1]

    def __iter__(self) -> Iterator[Tuple[str, array_like_1d]]:
        for column_name in self.column_names:
            yield column_name

    def __str__(self) -> str:
        # TODO: Make nice table string.
        return str(self.data)

    # popitem() is implemented by MutableMapping by default, but items are returned in
    # FIFO order. It is overridden to enforce LIFO order.
    def popitem(self) -> [str, np.ndarray]:
        """t.popitem() -> (k, v), remove and return some (key, value) pair
        as a 2-tuple; but raise KeyError if t is empty. Pairs removed in LIFO
        order."""
        try:
            key = self.column_names[-1]
        except IndexError:
            raise StopIteration
        value = self[key]
        del self[key]
        return key, value

    # Public methods
    def reset_index(self):
        if self.index is not None:
            if self._data is None:
                self._index = None
            else:
                self._index = np.arange(0, self.shape[0])

    # TODO: add_row function

    # TODO: iterrows iterator

    # Private Helper methods
    @staticmethod
    def _parse_data_dict(
        raw_data: Dict[str, array_like_1d]
    ) -> [np.ndarray, Dict[str, int]]:
        if len(raw_data) == 0:
            raise EmptyInputError()
        elif len(set([len(value) for value in raw_data.values()])) > 1:
            raise SizeMismatchError("Dictionary value sizes are not consistent.")

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
            raise EmptyInputError()

        return data

    def _set_index_from_data(self, raw_data: Optional[np.ndarray] = None):
        if raw_data is not None:
            raw_data = raw_data.reshape((-1, 1))
        else:
            raw_data = self.data
        self.index = np.arange(0, raw_data.shape[0])

    def _set_column_names_from_data(self):
        self.column_names = [f"Col{i}" for i in range(self.data.shape[1])]

    def _verify_key(self, key: str):
        if key not in (*self.column_names, self.index_name):
            raise KeyError(
                f"'{key}' not in column_names and '{key}' is not index_name."
            )


# Table Exceptions


class SizeMismatchError(Exception):
    """Exception that gets raised when the lengths of the
    index or columns do not match."""

    def __init__(self, context: Optional[str] = None) -> None:
        message = "Invalid input. Mis-matched data lengths."
        if context is not None:
            message += " " + context
        super().__init__(message)


class EmptyInputError(Exception):
    """Exception that is raised when inputs are or have an empty array."""

    def __init__(self) -> None:
        super().__init__("Input contains an empty array.")
