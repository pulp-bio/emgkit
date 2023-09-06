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

import numpy as np
from scipy import signal

from .._base import Signal, signal_to_array


def lowpass_filter(x: Signal, cut: float, fs: float, order: int = 2) -> np.ndarray:
    """Apply a Butterworth lowpass filter on the given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    cut : float
        Higher bound for frequency band.
    fs : float
        Sampling frequency.
    order : int, default=2
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(order, cut, btype="lowpass", output="sos", fs=fs)
    return signal.sosfiltfilt(sos, x_array, axis=0).copy()


def highpass_filter(x: Signal, cut: float, fs: float, order: int = 2) -> np.ndarray:
    """Apply a Butterworth highpass filter on the given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    cut : float
        Lower bound for frequency band.
    fs : float
        Sampling frequency.
    order : int, default=2
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(order, cut, btype="highpass", output="sos", fs=fs)
    return signal.sosfiltfilt(sos, x_array, axis=0).copy()


def bandpass_filter(
    x: Signal,
    low_cut: float,
    high_cut: float,
    fs: float,
    order: int = 2,
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
    order : int, default=2
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(
        order, (low_cut, high_cut), btype="bandpass", output="sos", fs=fs
    )
    return signal.sosfiltfilt(sos, x_array, axis=0).copy()


def bandstop_filter(
    x: Signal,
    low_cut: float,
    high_cut: float,
    fs: float,
    order: int = 2,
) -> np.ndarray:
    """Apply a Butterworth bandstop filter on the given signal.

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
    order : int, default=2
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered signal with shape (n_samples, n_channels).
    """
    # Convert input to array
    x_array = signal_to_array(x)
    # Create and apply filter
    sos = signal.butter(
        order, (low_cut, high_cut), btype="bandstop", output="sos", fs=fs
    )
    return signal.sosfiltfilt(sos, x_array, axis=0).copy()
