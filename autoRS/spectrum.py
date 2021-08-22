"""Response Spectra generation functions."""

# %% Import Necessary Modules

# Standard library imports
from __future__ import annotations
from time import perf_counter
from itertools import accumulate
from typing import Tuple, Union, List

# Third party imports
import numpy as np

# Type Aliases
numeric = Union[int, float]
array_like_1d = Union[List[numeric], Tuple[numeric], np.ndarray]

# %% Utility functions


def get_asme_frequencies() -> np.ndarray:
    """Generates array of frequencies (Hz) for spectral generation as specified
    in ASME B&PVC Division I, Section III, Non-mandatory appendix N, Table
    N-1226-1.

    Returns
    -------
    np.ndarray: Array of frequency points (Hz).
    """
    frq_range = (0.1, 3, 3.6, 5, 8, 15, 18, 22, 34)
    increments = (0.1, 0.15, 0.2, 0.25, 0.5, 1, 2, 3)
    frqs = np.concatenate(
        [np.arange(*args) for args in zip(frq_range[:-1], frq_range[1:], increments)]
    )
    frqs = np.append(frqs, frq_range[-1])
    return frqs


def get_default_frequencies(high_frequency: bool = False) -> np.ndarray:
    """Generates default array of frequencies (Hz), with 100 total points if the range
    is from [0.1Hz, 100Hz], or 115 total points if the range is [0.1Hz, 1000Hz].

    Parameters
    ----------
    high_frequency: bool
                    If false (default), frequency range is 100 points in [0.1Hz, 100Hz].
                    If true, frequency range is 115 total points in [0.1Hz, 1000Hz].

    Returns
    -------
    np.ndarray: Array of frequency points (Hz).
    """

    frqs = get_asme_frequencies()

    # Number of points to add from end of frqs to 100Hz
    npts_low = 100 - len(frqs)

    # Generate points up to 100Hz
    frqs = np.append(
        frqs,
        [
            round(x)
            for x in np.geomspace(frqs[-1], 100, npts_low + 1, endpoint=True)[1:]
        ],
    )

    if high_frequency:
        # Additional number of points from 100Hz to 1000Hz.
        npts_high = 15
        frqs = np.append(
            frqs,
            [
                round(x)
                for x in np.geomspace(frqs[-1], 1000, npts_high + 1, endpoint=True)[1:]
            ],
        )

    return frqs


# %% Raw private response spectrum generation functions


def _get_step_matrix(
    w: float, zeta: float, dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the A, B matrices from [1] based on the input
    angular frequency `w`, critical damping ratio, `zeta`,
    and timestep, `dt`.

    For use with the `step_resp_spect` function that generates
    response spectra by the step-by-step method.

    Parameters
    ----------
    w : float
        Angular frequency in rads/s.

    zeta : float
        Critical damping ratio (dimensionless).

    dt : float
        Timestep in s.

    Returns
    -------
    A : (2, 2) ndarray
        Array to be matrix-multiplied by the [x_i, xdot_i] vector.

    B : (2, 2) ndarray
        Array to be matrix-multiplied by the [a_i, a_(i+1)] vector.

    References
    ----------
    .. [1] Nigam, Jennings, April 1969. Calculation of response Sepctra
        from Stong-Motion Earthquake Records. Bulletin of the Seismological
        Society of America. Vol 59, no. 2.
    """

    A = np.zeros((2, 2))
    B = np.zeros((2, 2))

    exp = np.exp(-zeta * w * dt)
    zsqt = (1 - zeta ** 2) ** 0.5
    sin = np.sin(w * zsqt * dt)
    cos = np.cos(w * zsqt * dt)

    A[0, 0] = exp * (cos + sin * zeta / zsqt)
    A[0, 1] = exp / (w * zsqt) * sin
    A[1, 0] = -w / zsqt * exp * sin
    A[1, 1] = exp * (cos - sin * zeta / zsqt)

    t1 = (2 * zeta ** 2 - 1) / w ** 2 / dt
    t2 = 2 * zeta / w ** 3 / dt

    B[0, 0] = exp * (sin / (w * zsqt) * (t1 + zeta / w) + cos * (t2 + 1 / w ** 2)) - t2
    B[0, 1] = -exp * (sin / (w * zsqt) * t1 + cos * t2) - 1 / w ** 2 + t2
    B[1, 0] = (
        exp
        * (
            (t1 + zeta / w) * (cos - sin * zeta / zsqt)
            - (t2 + 1 / w ** 2) * (sin * w * zsqt + cos * zeta * w)
        )
        + 1 / w ** 2 / dt
    )
    B[1, 1] = (
        -exp * (t1 * (cos - sin * zeta / zsqt) - t2 * (sin * w * zsqt + cos * zeta * w))
        - 1 / w ** 2 / dt
    )

    return A, B


def _step_rs(
    acc: array_like_1d, time: array_like_1d, frqs: array_like_1d, zeta: float = 0.05,
) -> [np.ndarray, np.ndarray]:
    """Generate acceleration response spectrum by the step-by-step method [1].
    The algorithm matches the RS results from SHAKE2000. The theory behind
    the algorithm assumes a 'segmentally-linear' acceleration time history (TH).
    Hence, the implicit assumption is that the time history nyquist frequency is
    much higher than the highest frequency within the TH.

    Use the `_fft_rs` method if there is frequency content close to the
    nyquist frequency. Or, use `scipy.signal.resample` to up-sample the acc. TH
    prior to using `_step_rs`.

    Parameters
    ----------
    acc : 1d array_like
        Input 1D acceleration time history.
    time : 1d array_like
        Input 1D time values for the acceleration time history, `acc`.
    frqs : 1d array_like
        1D array of frequencies where the response is calculated.
    zeta : float, optional
        Critical damping ratio (dimensionless). Defaults to 0.05. Should be between 0
        and 1.

    Returns
    -------
    rs : ndarray
        Array with spectral accelerations (same units as input acc).
    frqs : ndarray
        Array with frequencies in Hz.

    References
    ----------
    .. [1] Nigam, Jennings, April 1969. Calculation of response Spectra
        from Strong-Motion Earthquake Records. Bulletin of the Seismological
        Society of America. Vol 59, no. 2.
    """

    # Enforce ndarray type
    frqs = np.array(frqs)

    # Instantiate angular frequency and spectral acceleration arrays
    w = frqs * 2 * np.pi
    rs = 0 * w

    # Define timestep from input signal
    dt = time[1] - time[0]

    # Calculate response for a spring with each wn
    for k, wn in enumerate(w):

        # Calculate response acceleration time history
        A, B = _get_step_matrix(wn, zeta, dt)

        # Define utility function to be used with itertools.accumulate
        def func(x_i: array_like_1d, a_i: array_like_1d) -> np.ndarray:
            return np.dot(A, x_i) + np.dot(B, a_i)

        act = np.column_stack((acc[:-1], acc[1:]))
        act = np.append(np.array([[0, 0], [0, acc[0]]]), act, axis=0)
        x = np.array(list(accumulate(act, func)))
        temp = -np.array([wn ** 2, 2 * zeta * wn])
        z = np.dot(x, temp)
        rs[k] = np.max(np.absolute(z))

    return rs, frqs


def _fft_rs(
    acc: array_like_1d, time: array_like_1d, frqs: array_like_1d, zeta: float = 0.05,
) -> [np.ndarray, np.ndarray]:
    """Generate acceleration response spectrum using a frequency domain
    method at the given frequencies. This is physically accurate if the true
    acceleration time history has no frequency content higher than the nyquist
    frequency of the input acceleration.

    Parameters
    ----------
    acc : 1d array_like
        Input 1D acceleration time history.
    time : 1d array_like
        Input 1D time values for the acceleration time history, `acc`.
    frqs : 1d array_like
        1D array of frequencies where the response is calculated.
    zeta : float, optional
        Critical damping ratio (dimensionless). Defaults to 0.05. Should be between 0
        and 1.

    Returns
    -------
    rs : ndarray
        Array with spectral accelerations (same units as input acc).
    frqs : ndarray
        Array with frequencies in Hz. Same as `frequencies`.
    """

    # Enforce ndarray type
    frqs = np.array(frqs)

    # Instantiate angular frequency and spectral acceleration arrays
    w = frqs * 2 * np.pi
    rs = 0 * w

    # Define minimum timestep from input signal
    dt_min = time[1] - time[0]

    # Calculate n, the integer to determine 0 padding at the end
    # of the time history; making n a power of 2 improves the
    # efficiency of the fft algorithm
    n = len(acc)
    n_fft = int(2 ** (np.ceil(np.log(1.5 * n) / np.log(2))))

    # Get n for upsampling by sinc-interpolating so there are
    # `multiplier` times as many points
    multiplier = 8

    # Get FFT of input acceleration
    xgfft = np.fft.rfft(acc, n_fft)
    frqt = np.fft.rfftfreq(n_fft, d=dt_min)

    # Calculate response for a spring with each wn
    for k, wn in enumerate(w):

        # Angular frequencies of fft
        wf = frqt * 2 * np.pi

        # Displacement of spring mass (fourier terms)
        xfft = -xgfft / (-(wf ** 2) + 2 * zeta * wn * 1j * wf + wn ** 2)

        # Relative acceleration of spring mass (fourier terms)
        accfft = -xfft * wf ** 2

        # Absolute acceleration of spring mass (fourier terms)
        abs_accfft = accfft + xgfft

        # Get absolute acceleration of spring mass (time domain)
        # Up-sample so that the final time history is sinc-
        # interpolated with `n_multiplier` total points
        a = np.fft.irfft(abs_accfft, n=multiplier * n_fft) * multiplier

        # Peak absolute acceleration of spring mass
        rs[k] = np.max(np.absolute(a))

    return rs, frqs


# %% Global Variables

# Constant that defines the available RS generation algorithms.
RS_METHODS_DICT = {
    "fft": _fft_rs,
    "shake": _step_rs,
}

RS_METHODS = tuple(RS_METHODS_DICT.keys())

DEFAULT_METHOD = "fft"


# %% Public RS generation functions


def response_spectrum(
    acc: array_like_1d,
    time: array_like_1d,
    zeta: float = 0.05,
    high_frequency: bool = False,
    method=DEFAULT_METHOD,
    # additional_frequencies: Optional[array_like_1d] = None,
    # verbose = True,
) -> [np.ndarray, np.ndarray]:
    """Generate acceleration response spectrum (RS) using one of the vailable methods.

    Parameters
    ----------
    acc : 1d array_like
        Input 1D acceleration time history.
    time : 1d array_like
        Input 1D time values for the acceleration time history, `acc`.
    zeta : float, optional, defalut = 0.05
        Critical damping ratio (dimensionless). Defaults to 0.05. Should be between 0
        and 1.
    high_frequency : bool, optional, default = False
        Boolean that determines frequency range of RS. If false, the range is
        [0.1Hz, 100Hz]. If true, the range is [0.1Hz, 1000Hz].
    method : str, optional, default = `DEFAULT_METHOD`
        The RS method to be used. See `RS_METHODS`.

    Returns
    -------
    rs : ndarray
        Array with spectral accelerations (same units as input acc).
    frqs : ndarray
        Array with frequencies in Hz.
    """

    rs_func = RS_METHODS_DICT.get(method, RS_METHODS_DICT[DEFAULT_METHOD])

    # Start timer
    t0 = perf_counter()

    # Get array of frequencies for RS calculation
    frqs = get_default_frequencies(high_frequency=high_frequency)

    # Run RS algorithm
    rs, _ = rs_func(acc, time, frqs, zeta)

    # End timer and print timing info
    t1 = perf_counter()
    t_net = t1 - t0

    print(
        "RS done. Time taken = {:.5f}s".format(t_net),
        "\ntime per iteration = {:.5f}s".format(t_net / len(frqs)),
    )

    return rs, frqs
