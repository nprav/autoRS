"""Miscellaneous utility functions for the autoRS unit-tests."""

# Standard library imports
from typing import Optional

# Third party imports
import numpy as np

# Local Application Imports
# from context import autoRS
from autoRS.typing import array_like_1d, numeric


# Get closest higher power of 2
def _next_pwr_of_2(n: numeric) -> int:
    """Get the closest higher number that is a power of 2"""
    pwr = np.ceil(np.log(n) / np.log(2))
    return int(2 ** pwr)


# Hard low-pass filter (used in test cases)
def low_pass_filter(
    signal: array_like_1d,
    lp_frq: numeric,
    dt: float = 0.005,
    time: array_like_1d = None,
    zero_pad: bool = True,
) -> np.ndarray:
    """Function to low-pass filter a signal in the frequency domain
    by setting all fourier terms larger than `lp_freq` to 0.
    """
    if time is not None:
        dt = time[1] - time[0]
    if zero_pad:
        n = 2 * _next_pwr_of_2(len(signal))
    else:
        n = len(signal)
    frq = np.fft.rfftfreq(n, d=dt)
    signal_fft = np.fft.rfft(signal, n=n)
    lp_fft = signal_fft * (frq <= lp_frq)
    lp_signal = np.fft.irfft(lp_fft, n)[: len(signal)]
    return lp_signal
