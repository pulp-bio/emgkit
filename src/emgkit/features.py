"""
This module contains functions for computing EMG features.


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

from ._base import Signal, signal_to_array

__all__ = [
    "root_mean_square",
    "waveform_length",
]


def waveform_length(x: Signal, win_len_ms: float, fs: float) -> np.ndarray:
    """
    Compute the waveform length of a given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    win_len_ms : float
        Window length (in ms).
    fs : float
        Sampling frequency.

    Returns
    -------
    ndarray
        Waveform length of the signal.
    """

    # Convert to array
    x_array = signal_to_array(x, allow_1d=True)

    win_len = int(round(win_len_ms / 1000 * fs))
    abs_diff = np.abs(np.diff(x_array, axis=0, prepend=0))
    kernel = np.ones(win_len)
    wl = np.stack(
        [
            signal.convolve(abs_diff[:, i], kernel, mode="valid")
            for i in range(x_array.shape[1])
        ],
        axis=1,
    )
    return wl


def root_mean_square(x: Signal, win_len_ms: float, fs: float) -> np.ndarray:
    """
    Compute the RMS of a given signal.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    win_len_ms : float
        Window length (in ms).
    fs : float
        Sampling frequency.

    Returns
    -------
    ndarray
        RMS of the signal.
    """

    # Convert to array
    x_array = signal_to_array(x, allow_1d=True)

    win_len = int(round(win_len_ms / 1000 * fs))
    x_sq = x_array**2
    kernel = np.ones(win_len) / win_len
    rms = np.stack(
        [
            np.sqrt(signal.convolve(x_sq[:, i], kernel, mode="valid"))
            for i in range(x_array.shape[1])
        ],
        axis=1,
    )
    return rms
