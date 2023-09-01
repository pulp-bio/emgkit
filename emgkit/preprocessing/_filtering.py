"""Functions implementing filters.


Copyright 2023 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal

from .._base import Signal, signal_to_array


def lowpass_filter(x: Signal, cut: float, fs: float, order: int = 5) -> np.ndarray:
    """Apply a Butterworth lowpass filter on the given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    cut : float
        Higher bound for frequency band.
    fs : float
        Sampling frequency.
    order : int, default=5
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(order, cut, btype="low", output="sos", fs=fs)
    return signal.sosfiltfilt(sos, x_array, axis=0).copy()


def highpass_filter(x: Signal, cut: float, fs: float, order: int = 5) -> np.ndarray:
    """Apply a Butterworth highpass filter on the given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    cut : float
        Lower bound for frequency band.
    fs : float
        Sampling frequency.
    order : int, default=5
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(order, cut, btype="high", output="sos", fs=fs)
    return signal.sosfiltfilt(sos, x_array, axis=0).copy()


def bandpass_filter(
    x: Signal,
    low_cut: float,
    high_cut: float,
    fs: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter on the given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    low_cut : float
        Lower bound for frequency band.
    high_cut : float
        Higher bound for frequency band.
    fs : float
        Sampling frequency.
    order : int, default=5
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(order, (low_cut, high_cut), btype="band", output="sos", fs=fs)
    return signal.sosfiltfilt(sos, x_array, axis=0).copy()


def notch_filter(
    x: Signal,
    exclude_freqs: Sequence[float],
    fs: float,
    exclude_harmonics: bool = False,
    max_harmonic: float | None = None,
    q: float = 30.0,
) -> np.ndarray:
    """Apply a notch filter on the given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    exclude_freqs : sequence of floats
        Frequencies to exclude.
    fs : float
        Sampling frequency.
    exclude_harmonics : bool, default=False
        Whether to exclude all the harmonics, too.
    max_harmonic : float or None, default=None
        Maximum harmonic to exclude.
    q : float, default=30.0
        Quality factor of the filters.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)

    def find_multiples(base: float, limit: float) -> list[float]:
        last_mult = int(round(limit / base))
        return [base * i for i in range(1, last_mult + 1)]

    # Find harmonics, if required
    if exclude_harmonics:
        if max_harmonic is None:
            max_harmonic = fs // 2
        exclude_freqs_set = set(
            [f2 for f1 in exclude_freqs for f2 in find_multiples(f1, max_harmonic)]
        )
    else:
        exclude_freqs_set = set(exclude_freqs)

    # Apply series of notch filters
    for freq in exclude_freqs_set:
        b, a = signal.iirnotch(freq, q, fs)
        x_array = signal.filtfilt(b, a, x_array, axis=0)

    return x_array.copy()
