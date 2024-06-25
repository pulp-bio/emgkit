"""
This module contains functions for computing statistics of spike trains.


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
import pandas as pd
from scipy import signal

from ._base import Signal, signal_to_array

__all__ = [
    "cov_amp",
    "cov_isi",
    "instantaneous_discharge_rate",
    "smoothed_discharge_rate",
]


def instantaneous_discharge_rate(spikes_t: np.ndarray) -> pd.Series:
    """
    Compute the instantaneous discharge rate from the given spike train.

    Parameters
    ----------
    spikes_t : ndarray
        Array containing the time of spikes (in seconds).

    Returns
    -------
    Series
        A Series containing the instantaneous discharge rate.
    """
    dr = 1 / np.diff(spikes_t)
    return pd.Series(dr, index=spikes_t[1:])


def smoothed_discharge_rate(
    spikes_bin: Signal, fs: float, win_len_s: float = 1.0
) -> pd.Series:
    """
    Compute the smoothed discharge rate from the spike train using a Hanning window.

    Parameters
    ----------
    spikes_bin : Signal
        Binary representation of the spike train with shape (n_samples,)
        containing either ones or zeros (spike/not spike).
    fs : float
        Sampling frequency.
    win_len_s : float, default=1.0
        Size (in seconds) of the Hanning window.

    Returns
    -------
    Series
        A Series containing the smoothed discharge rate.
    """
    # Convert to array
    spikes_bin_array = signal_to_array(spikes_bin, allow_1d=True).flatten()

    win_len = int(win_len_s * fs)
    win = signal.windows.hann(win_len)
    win = win / win.sum() * fs  # normalize window area
    dr = signal.convolve(spikes_bin_array, win, mode="same")
    return pd.Series(dr, index=np.arange(dr.size) / fs)


def cov_isi(spikes_t: np.ndarray) -> float:
    """
    Compute the Coefficient of Variation of the Inter-Spike Interval (CoV-ISI) of the given spike train.

    Parameters
    ----------
    spikes_t : ndarray
        Array containing the time of spikes (in seconds).

    Returns
    -------
    float
        The CoV-ISI of the spike train.
    """
    # Compute ISI
    isi = np.diff(spikes_t)

    res = np.nan
    if isi.size > 1:
        # Compute CoV-ISI
        res = isi.std() / isi.mean()
    return res


def cov_amp(spikes_amp: np.ndarray) -> float:
    """
    Compute the Coefficient of Variation of the spike amplitude of the given spike train.

    Parameters
    ----------
    spikes_amp : ndarray
        Array containing the amplitude of spikes.

    Returns
    -------
    float
        The CoV-Amp of the spike train.
    """
    res = np.nan
    if spikes_amp.size > 1:
        # Compute CoV-Amp
        res = spikes_amp.std() / spikes_amp.mean()
    return res
