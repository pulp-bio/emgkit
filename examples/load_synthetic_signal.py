"""Functions to load and pre-process the synthetic sEMG dataset used by
Mohebian et al. in their work (https://doi.org/10.3389/fncom.2019.00014).


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

import glob
import os
from functools import reduce

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal


def _band_limited_noise(
    min_freq: int, max_freq: int, n_samples: int, fs: int = 1, seed: int | None = None
) -> np.ndarray:
    """Generate band-limited white Gaussian noise."""
    t = np.linspace(0, n_samples / fs, n_samples)
    freqs = np.arange(min_freq, max_freq + 1, n_samples / fs)
    phases = np.random.default_rng(seed).random(len(freqs)) * 2 * np.pi
    s = [np.sin(2 * np.pi * freq * t + phase) for freq, phase in zip(freqs, phases)]
    s = reduce(lambda a, b: a + b, s)
    s /= np.max(s)
    return s


def load_synthetic_signal(
    data_path: str,
    mvc: int,
    snr: int | None = None,
    apply_filter: bool = True,
    seed: int | None = None,
) -> list[tuple[pd.DataFrame, dict[int, np.ndarray], float]]:
    """Load data from the simulated dataset given the MVC value.

    Parameters
    ----------
    data_path : str
        Path to the dataset root folder.
    mvc : {10, 30, 50}
        Effort level as percentage of MVC.
    snr : int or None, default=None
        Amount of noise in the bandwidth of 20-500 Hz to add to the signal.
    apply_filter: bool, default=True
        Whether the signals should be filtered or not.
    seed: int or None, default=None
        Random seed for reproducibility (relevant if snr is not None).

    Returns
    -------
    list of tuples of (ndarray, dict of {int : ndarray}, float)
        List containing, for each simulation, a tuple which comprises:
        - the sEMG signal as a DataFrame with shape (n_samples, n_channels);
        - the ground-truth spike times as a dictionary;
        - the sampling frequency.
    """
    assert mvc in [10, 30, 50], "The MVC value type must be either 10, 30 or 50."

    data = []
    for path in sorted(glob.glob(os.path.join(data_path, f"S*_{mvc}MVC.mat"))):
        cur_data = sio.loadmat(path)

        # Load sEMG data
        tmp = cur_data["sig_out"]
        n_ch = tmp.shape[0] * tmp.shape[1]
        n_samp = tmp[0, 0].shape[1]
        emg = np.zeros(shape=(n_samp, n_ch))
        k = 0
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                emg[:, k] = tmp[i, j]
                k += 1

        # Load sampling frequency
        fs = int(cur_data["fsamp"].item())

        # Load ground-truth spike trains
        n_mu = cur_data["sFirings"].shape[1]
        gt_spikes = {}
        for i in range(n_mu):
            cur_discharges = cur_data["sFirings"][0, i].flatten() / fs
            gt_spikes[i] = cur_discharges

        # Apply noise, if specified
        if snr is not None:
            # Compute signal power and convert to dB
            emg_avg_power = np.mean(np.square(emg), axis=0)
            emg_avg_db = 10 * np.log10(emg_avg_power)
            # Compute noise power
            noise_avg_db = emg_avg_db - snr
            noise_avg_power = 10 ** (noise_avg_db / 10)

            # Generate 20-500Hz noise with given power
            noise = np.zeros_like(emg)
            for i in range(n_ch):
                noise[:, i] = _band_limited_noise(
                    min_freq=20, max_freq=500, n_samples=n_samp, fs=fs, seed=seed
                )
                noise[:, i] *= np.sqrt(noise_avg_power[i])

            # Noise up the original signal
            emg += noise

            # Apply 1st order, 20-500Hz Butterworth filter, if specified
            if apply_filter:
                sos = signal.butter(
                    N=1, Wn=(20, 500), btype="bandpass", output="sos", fs=fs
                )
                emg = signal.sosfiltfilt(sos, emg, axis=0).copy()

        # Pack in DataFrame
        emg = pd.DataFrame(
            data=emg,
            index=np.arange(n_samp) / fs,
            columns=[f"EMG_{i + 1}" for i in range(n_ch)],
        )
        data.append((emg, gt_spikes, fs))

    return data
