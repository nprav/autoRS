"""Defines custom classes for representing Motion Time Histories and
Motion Fourier Spectra"""
# %% Import Necessary Modules

# Standard library imports
from typing import Tuple, Optional, Union, Collection
from abc import ABC, abstractmethod
from copy import deepcopy, copy

# Third party imports
import numpy as np

# Local application imports
from autoRS.typing import array_like_1d
from autoRS.core.table import Table, FloatTable
from autoRS.core.utils import cumulative_integral

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

    @property
    def w(self) -> np.ndarray:
        return 2 * np.pi * self.frequency


# %% Main Class definitions

# TODO: Add a TableIloc/TableView class that can be used to set
#   act on specific parts of MotionTable components similar to np.ndarray
#   indexing.


class MotionHistory(MotionTable, TimeHistory):
    """Class representing a motion time history, including acceleration, velocity,
    displacement, and an index with the time values."""

    TableClass = FloatTable

    def __init__(
        self,
        time: Optional[array_like_1d] = None,
        acceleration: Optional[array_like_1d] = None,
        velocity: Optional[array_like_1d] = None,
        displacement: Optional[array_like_1d] = None,
        dt: Optional[float] = None,
    ) -> None:

        # Initialize table and other private variables
        self.reset()
        self._last_updated: Optional[None] = None

        # Setup motion inputs
        if acceleration is not None:
            self.acceleration = np.asarray(acceleration)
        elif velocity is not None:
            self.velocity = np.asarray(velocity)
        elif displacement is not None:
            self.displacement = np.asarray(displacement)

        # Setup time inputs
        if time is not None:
            self.time = np.asarray(time)
        elif dt is not None and self._table.shape[0] > 0:
            self.time_from_dt(dt)
        elif dt is not None:
            raise ValueError(
                "No acceleration/velocity/displacement defined. dt is ignored."
            )

    @property
    def table(self) -> TableClass:
        return deepcopy(self._table)

    @property
    def time(self) -> np.ndarray:
        return copy(self._table.index)

    @time.setter
    def time(self, value: array_like_1d) -> None:
        value = np.asarray(value)
        self._table.index = value

    @property
    def dt(self) -> Union[np.ndarray, float, None]:
        if self.time is None:
            return None
        diff_t = np.diff(self.time)
        if len(np.unique(diff_t)) != 0:
            return diff_t[0]
        else:
            return diff_t

    @property
    def acceleration(self) -> np.ndarray:
        if np.isnan(self._table["acceleration"][0]) and self._last_updated is not None:
            self._table["acceleration"] = np.gradient(self.velocity, self.time)
        return copy(self._table["acceleration"])

    @acceleration.setter
    def acceleration(self, value: array_like_1d) -> None:
        self._table["acceleration"] = value
        self._last_updated = "acceleration"
        self.reset(skip_columns=("time", "acceleration"))

    @property
    def velocity(self) -> np.ndarray:
        if np.isnan(self._table["velocity"][0]):
            if self._last_updated == "displacement":
                self._table["velocity"] = np.gradient(self.displacement, self.time)
            elif self._last_updated == "acceleration":
                self._table["velocity"] = cumulative_integral(
                    self.acceleration, self.time, method="trapezoidal",
                )
        return copy(self._table["velocity"])

    @velocity.setter
    def velocity(self, value: array_like_1d) -> None:
        self._table["velocity"] = value
        self._last_updated = "velocity"
        self.reset(skip_columns=("time", "velocity"))

    @property
    def displacement(self) -> np.ndarray:
        if np.isnan(self._table["displacement"][0]) and self._last_updated is not None:
            self._table["displacement"] = cumulative_integral(
                self.velocity, self.time, method="trapezoidal",
            )
        return copy(self._table["displacement"])

    @displacement.setter
    def displacement(self, value: array_like_1d) -> None:
        self._table["displacement"] = value
        self._last_updated = "displacement"
        self.reset(skip_columns=("time", "displacement"))

    def time_from_dt(self, dt: float, npts: Optional[int] = None):
        if npts is None:
            if self._table.shape[0] == 0:
                raise ValueError(
                    "npts is undefined. Must either define one of acceleration, "
                    "velocity, and displacement, or input a value for npts."
                )
            else:
                npts = self._table.shape[0]
        self.time = np.arange(0, npts * dt, dt)

    def reset(self, skip_columns: Optional[Collection[str]] = None) -> None:
        column_set = {"time", "acceleration", "velocity", "displacement"}
        if skip_columns is None:
            self._table = self.TableClass(
                column_names=self._motion_column_names, index_name=self._index_name,
            )
        else:
            for col in column_set - set(skip_columns):
                self._table[col] = np.ones(self._table.shape[0]) * np.nan


class MotionSpectra(MotionTable, Spectrum):
    """Class representing a motion spectrum, including acceleration, velocity,
    displacement, and an index with frequencies."""

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
