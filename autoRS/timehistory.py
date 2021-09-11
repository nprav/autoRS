"""Defines a custom class for representing Time Histories."""
# %% Import Necessary Modules

# Standard library imports
from itertools import accumulate
from typing import Tuple, Union, List, Optional

# Third party imports
import numpy as np

# Local application imports
from autoRS import numeric, array_like_1d

# %% Class definition


class TimeHistory:
    """Class representing a motion time history, including acceleration, velocity,
    displacement, and their fourier transforms."""

    _motion_vars = ("acceleration", "velocity", "displacement")

    def __init__(
        self, dt: numeric, **kwargs,
    ):
        self.time_vars = {
            key: np.array(kwargs[key]) if key in kwargs else None
            for key in self._motion_vars
        }
        if not self.time_vars:
            raise ValueError(
                "At least one of acceleration, velocity, or displacement"
                "must be input to define the TimeHistory."
            )
        self.defined_key = list(self.time_vars.keys())[0]
        self.npts = len(self.time_vars[self.defined_key])
        self.dt = dt
        self.time = np.linspace(0, self.dt * self.npts, num=self.npts)
        self.frequencies = None
        self.fourier_vars = {key: None for key in self._motion_vars}
