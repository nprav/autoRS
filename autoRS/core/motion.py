"""Defines a custom class for representing Time Histories."""
# %% Import Necessary Modules

# Standard library imports
from typing import Tuple
from abc import ABC, abstractmethod
from copy import deepcopy

# Third party imports
import numpy as np

# Local application imports
from autoRS.typing import numeric, array_like_1d
from .table import Table, FloatTable

# %% Abstract Class definitions


class MotionTable(ABC):
    """An interface for a Table that describes the motion of an entity. The class
    must describe displacements, velocities, and accelerations of an object."""

    _motion_column_names: Tuple[str] = ("acceleration", "velocity", "displacement")

    @property
    @abstractmethod
    def table(self) -> Table:
        pass

    @property
    @abstractmethod
    def acceleration(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def velocity(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def displacement(self) -> np.ndarray:
        pass


class TimeHistory(ABC):
    """Abstract class for Tables that represent Time Histories. The class must
    use a time index."""

    _index_name: str = "time"

    @property
    @abstractmethod
    def time(self) -> np.ndarray:
        pass


class Spectrum(ABC):
    """Abstract class for Tables that represent spectra. The class must
    use a frequency index."""

    _index_name: str = "frequency"

    @property
    @abstractmethod
    def frequency(self) -> np.ndarray:
        pass


# %% Abstract Class definitions


class MotionHistory(MotionTable, TimeHistory):
    """Class representing a motion time history, including acceleration, velocity,
    displacement, and their fourier transforms."""

    TableClass = FloatTable

    def __init__(
        self,
        time: array_like_1d = None,
        dt: numeric = None,
        acceleration: array_like_1d = None,
        velocity: array_like_1d = None,
        displacement: array_like_1d = None,
    ) -> None:

        # Initialize table
        self._table = FloatTable(
            column_names=self._motion_column_names, index_name=self._index_name
        )

        # Setup motion inputs
        if acceleration is not None:
            self.acceleration = np.asarray(acceleration)

        # Setup time inputs
        if time is not None:
            self._table.index = np.asarray(time)
            self.dt = time[1] - time[0]

    @property
    def time(self) -> np.ndarray:
        return self._table.index

    @time.setter
    def time(self, value: np.ndarray) -> None:
        self._table.index = value

    @property
    def table(self) -> TableClass:
        return deepcopy(self._table)

    @property
    def acceleration(self) -> np.ndarray:
        return self._table["acceleration"]

    @acceleration.setter
    def acceleration(self, value: array_like_1d) -> None:
        self._table["acceleration"] = value

    @property
    def velocity(self) -> np.ndarray:
        return self._table["velocity"]

    @velocity.setter
    def velocity(self, value: array_like_1d) -> None:
        self._table["velocity"] = value

    @property
    def displacement(self) -> np.ndarray:
        return self._table["displacement"]

    @displacement.setter
    def displacement(self, value: array_like_1d) -> None:
        self._table["displacement"] = value


class MotionSpectra(MotionTable, Spectrum):

    TableClass = FloatTable

    def __init__(self) -> None:
        pass

    @property
    def frequency(self) -> np.ndarray:
        return self.table.index

    @property
    def table(self) -> TableClass:
        return self.TableClass()

    @property
    def acceleration(self) -> np.ndarray:
        pass

    @property
    def velocity(self) -> np.ndarray:
        pass

    @property
    def displacement(self) -> np.ndarray:
        pass
