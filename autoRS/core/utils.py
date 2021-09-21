"""Miscellaneous functions used in autoRS data structures."""


# %% Import libraries

# Standard library imports
from typing import Dict, Callable, Optional

# Third-party imports
import numpy as np

# Local application imports
from autoRS.typing import array_like_1d

# %% Main functions


def _cumulative_trapezoidal(
    y: np.ndarray, x: np.ndarray, initial: float,
) -> np.ndarray:
    """Trapezoidal integration of samples."""

    sub_integrals = np.diff(x) * (y[:-1] + y[1:]) / 2

    return np.cumsum(np.concatenate(([initial], sub_integrals)))


def _cumulative_simpson(y: np.ndarray, x: np.ndarray, initial: float,) -> np.ndarray:
    """1/3 composite simpson's integration of samples."""

    res = [initial]

    # Trapezoidal rule for the first integration point
    value = (x[1] - x[0]) * (y[1] + y[0]) / 2
    res.append(res[-1] + value)

    if len(y) == 2:
        return np.asarray(res)

    # 1/3 Simpson's rule for all remaining value
    sub_integrals = (x[2:] - x[:-2]) / 6 * (y[:-2] + 4 * y[1:-1] + y[2:])
    for value in sub_integrals:
        res.append(res[-2] + value)

    return np.asarray(res)


# Dictionary of integration options
integration_methods: Dict[str, Callable] = {
    "trapezoidal": _cumulative_trapezoidal,
    "simpson 1/3": _cumulative_simpson,
}


def cumulative_integral(
    y: array_like_1d,
    x: Optional[array_like_1d] = None,
    dx: float = 1,
    initial: Optional[float] = 0,
    method: str = "simpson 1/3",
) -> np.ndarray:
    """Numerical integration of input sequence (y)
    over an optional sequence of x values. Available methods are read from the
    `integration_methods` dictionary. The 1/3 composite simpson's rule is the default.
    The trapezoidal integration scheme is also available. This should
    be used there is significant frequency content that is more than 0.33x the
    signal sample rate."""

    if method not in integration_methods.keys():
        method = "simpson 1/3"

    y = np.asarray(y)

    if len(y) == 1:
        raise ValueError(
            "y must have 2 or more elements to be able to perform integration"
        )

    if x is not None:
        x = np.asarray(x)
        if len(x) != len(y):
            raise ValueError("y and x must have the same length.")
    else:
        x = np.arange(0, dx * len(y), dx)

    return integration_methods[method](y, x, initial)
