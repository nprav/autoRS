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
    y: np.ndarray, x: Optional[np.ndarray] = None, dx: float = 1, initial: float = 0,
) -> np.ndarray:
    """Trapezoidal integration of samples."""

    if x is not None:
        sub_integrals = np.diff(x) * (y[:-1] + y[1:]) / 2
    else:
        sub_integrals = dx * (y[:-1] + y[1:]) / 2

    return np.cumsum(np.concatenate(([initial], sub_integrals)))


def _cumulative_simpson_quad(
    y: np.ndarray, dx: float = 1, initial: float = 0,
) -> np.ndarray:
    """1/3 composite simpson's integration of samples.
    Assumes equally-spaced intervals. Assumes length of y > 3."""

    sub_integrals_fwd = dx / 3 * (5 * y[:-2] / 4 + 2 * y[1:-1] - y[2:] / 4)
    sub_integrals_bwd = dx / 3 * (-y[:-2] / 4 + 2 * y[1:-1] + 5 * y[2:] / 4)

    sub_integrals = np.concatenate(
        (
            [initial],
            [sub_integrals_fwd[0]],
            (sub_integrals_fwd[1:] + sub_integrals_bwd[:-1]) / 2,
            [sub_integrals_bwd[-1]],
        )
    )

    return np.cumsum(sub_integrals)


def _cumulative_simpson_cubic(
    y: np.ndarray, dx: float = 1, initial: float = 0,
) -> np.ndarray:
    """3/8 composite simpson's integration of samples.
    Assumes equally-spaced intervals. Assumes length of y > 4."""

    sub_integrals_fwd = dx * (
        3 * y[:-3] / 8 + 19 * y[1:-2] / 24 - 5 * y[2:-1] / 24 + y[3:] / 24
    )
    sub_integrals_mid = dx * (
        (-y[:-3] / 24 + 13 * y[1:-2] / 24 + 13 * y[2:-1] / 24 - y[3:] / 24)
    )
    sub_integrals_bwd = dx * (
        (y[:-3] / 24 - 5 * y[1:-2] / 24 + 19 * y[2:-1] / 24 + 3 * y[3:] / 8)
    )

    sub_integrals = np.concatenate(
        (
            [initial],
            [sub_integrals_fwd[0]],
            [(sub_integrals_fwd[1] + sub_integrals_mid[0]) / 2],
            (sub_integrals_fwd[2:] + sub_integrals_mid[1:-1] + sub_integrals_bwd[:-2])
            / 3,
            [(sub_integrals_bwd[-2] + sub_integrals_mid[-1]) / 2],
            [sub_integrals_bwd[-1]],
        )
    )

    return np.cumsum(sub_integrals)


# Dictionary of integration options
integration_methods: Dict[str, Callable] = {
    "trapezoidal": _cumulative_trapezoidal,
    "simpson 1/3": _cumulative_simpson_quad,
    "simpson 3/8": _cumulative_simpson_cubic,
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
    The trapezoidal and 3/8 composite simpson's rule integration schemes are also
     available."""

    if method not in integration_methods.keys():
        method = "simpson 1/3"

    y = np.asarray(y)

    if len(y) == 1:
        raise ValueError(
            "y must have 2 or more elements to be able to perform integration"
        )
    elif len(y) == 2:
        method = "trapezoidal"
    elif len(y) == 3 and method == "simpson 3/8":
        method = "simpson 1/3"

    if x is not None:
        x = np.asarray(x)
        if len(x) != len(y):
            raise ValueError("y and x must have the same length.")
        if np.std(np.diff(x)) > 0.02 * np.mean(np.diff(x)):
            return integration_methods["trapezoidal"](y, x=x, initial=initial)
        else:
            dx = np.mean(np.diff(x))

    return integration_methods[method](y, dx=dx, initial=initial)
