"""Class implementing the MU tracking algorithm.


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
import torch
from scipy import signal

from .._base import Signal, signal_to_array, signal_to_tensor
from ..preprocessing import extend_signal


class MUTracker:
    """Class implementing the MU tracking algorithm based on
    the Bell-Sejnowski algorithm with natural gradient.

    Parameters
    ----------
    fs : float
        Sampling frequency of the signal.
    f_ext : int
        Extension factor (in samples).
    mean_vec_init : ndarray or Tensor
        Initial mean vector with shape (n_channels,).
    white_mtx_init : ndarray or Tensor
        Initial whitening matrix with shape (n_pcs, n_channels).
    sep_mtx_init : ndarray or Tensor
        Initial separation matrix with shape (n_mu, n_pcs).
    spike_ths_init : ndarray
        Initial array with shape (n_mu,) containing the spike/noise thresholds.
    learning_rate : float, default=0.01
        Learning rate.
    momentum: float, default=0.9
        Momentum.
    n_gd_steps : int, default=1
        Number of steps of gradient descent.
    device : device or str, default="cpu"
        Torch device.

    Attributes
    ----------
    _fs : float
        Sampling frequency of the signal.
    _f_ext : int
        Extension factor.
    _ext_buf : ndarray or None
        Buffer for the on-line signal extension.
    _mean_vec : Tensor
        Mean vector with shape (n_channels,).
    _white_mtx : Tensor
        Whitening matrix with shape (n_pcs, n_channels).
    _sep_mtx : Tensor
        Separation matrix with shape (n_mu, n_pcs).
    _learning_rate : float
        Learning rate.
    _momentum : float
        Momentum.
    _white_vel : float
        Velocity term for whitening optimization.
    _sep_vel : float
        Velocity term for separation optimization.
    _n_gd_steps : int
        Number of steps of gradient descent.
    _sl : list of float
        Running estimate of spike level for each MU in the BSS output before integration.
    _nl : list of float
        Running estimate of noise level for each MU in the BSS output before integration.
    _n_samp_seen : int
        Number of samples seen.
    _device : device
        Torch device.
    """

    def __init__(
        self,
        fs: float,
        f_ext: int,
        mean_vec_init: np.ndarray | torch.Tensor,
        white_mtx_init: np.ndarray | torch.Tensor,
        sep_mtx_init: np.ndarray | torch.Tensor,
        spike_ths_init: np.ndarray,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        n_gd_steps: int = 1,
        device: torch.device | str = "cpu",
    ) -> None:
        self._fs = fs

        # Pre-processing
        self._f_ext = f_ext
        self._ext_buf = None

        # Whitening and BSS
        self._device = torch.device(device) if isinstance(device, str) else device
        self._sep_mtx = (
            torch.tensor(sep_mtx_init).to(self._device)
            if isinstance(sep_mtx_init, np.ndarray)
            else sep_mtx_init.clone().to(self._device)
        )
        self._white_mtx = (
            torch.tensor(white_mtx_init).to(self._device)
            if isinstance(white_mtx_init, np.ndarray)
            else white_mtx_init.clone().to(self._device)
        )
        self._mean_vec = (
            torch.tensor(mean_vec_init).to(self._device)
            if isinstance(mean_vec_init, np.ndarray)
            else mean_vec_init.clone().to(self._device)
        )
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._white_vel = 0
        self._sep_vel = 0
        self._n_gd_steps = n_gd_steps

        # Spike detection
        self._sl = 2 * spike_ths_init.copy()
        self._nl = np.zeros(self._sep_mtx.size(0), dtype=spike_ths_init.dtype)
        self._sl_hist = np.zeros(
            shape=(0, self._sep_mtx.size(0)), dtype=spike_ths_init.dtype
        )
        self._nl_hist = np.zeros(
            shape=(0, self._sep_mtx.size(0)), dtype=spike_ths_init.dtype
        )
        self._n_samp_seen = 0

    @property
    def sl_hist(self) -> np.ndarray:
        """ndarray: Property representing the history of the signal level."""
        return self._sl_hist

    @property
    def nl_hist(self) -> np.ndarray:
        """ndarray: Property representing the history of the noise level."""
        return self._nl_hist

    @property
    def th_hist(self) -> np.ndarray:
        """ndarray: Property representing the history of the threshold."""
        return self._nl_hist + 0.5 * (self._sl_hist - self._nl_hist)

    def process_window(self, emg: Signal) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Process a window and adapt internal parameters.

        Parameters
        ----------
        emg : Signal
            A window with shape (n_samples, n_channels).

        Returns
        -------
        DataFrame
            A DataFrame with shape (n_samples, n_mu) containing the components estimated by ICA.
        dict of {str : ndarray}
            Dictionary containing the discharge times for each MU.
        """
        # Convert to array
        emg_array = signal_to_array(emg)

        # Extension
        if self._ext_buf is not None:
            emg_array = np.concatenate(
                (self._ext_buf, emg_array), dtype=emg_array.dtype
            )
        emg_ext = extend_signal(emg_array, self._f_ext)
        self._ext_buf = emg_array[-self._f_ext + 1 :].copy()
        n_samp = emg_ext.shape[0]

        # Convert to Tensor
        emg_tensor = signal_to_tensor(emg_ext, self._device).T

        # Centering
        mean_vec_new = emg_tensor.mean(dim=1, keepdim=True)
        self._mean_vec = 0.5 * self._mean_vec + 0.5 * mean_vec_new
        emg_tensor -= self._mean_vec

        # On-line whitening
        emg_white = self._white_mtx @ emg_tensor
        for _ in range(self._n_gd_steps):
            delta_w = self._learning_rate * (
                self._white_mtx
                - emg_white @ emg_white.T / (n_samp - 1) @ self._white_mtx
            )
            self._white_vel = self._momentum * self._white_vel + delta_w
            self._white_mtx += self._white_vel
            emg_white = self._white_mtx @ emg_tensor

        # On-line BSS
        ics = self._sep_mtx @ emg_white
        for _ in range(self._n_gd_steps):
            delta_w = self._learning_rate * (
                self._sep_mtx
                - 2 * torch.tanh(ics) @ ics.T / (n_samp - 1) @ self._sep_mtx
            )
            self._sep_vel = self._momentum * self._sep_vel + delta_w
            self._sep_mtx += self._sep_vel
            ics = self._sep_mtx @ emg_white

        # Spike detection
        spikes_t = {}
        ics_array = ics.cpu().numpy()
        for i, ic_i in enumerate(ics_array):  # iterate over MUs
            spikes_t[f"MU{i}"] = np.asarray([], dtype=np.float32)

            # Detect peaks, and compare each one agains signal and noise estimates
            peaks, _ = signal.find_peaks(
                ic_i, height=0, distance=int(round(20e-3 * self._fs))
            )
            for peak_idx in peaks:
                peak_val = ic_i[peak_idx]

                th = self._nl[i] + 0.5 * (self._sl[i] - self._nl[i])
                if peak_val < th:
                    self._nl[i] = 0.2 * peak_val + 0.8 * self._nl[i]
                    continue

                self._sl[i] = 0.2 * peak_val + 0.8 * self._sl[i]

                # Save peak as spike
                spikes_t[f"MU{i}"] = np.append(
                    spikes_t[f"MU{i}"], (peak_idx + self._n_samp_seen) / self._fs
                )

        # Update history
        self._sl_hist = np.concatenate(
            (self._sl_hist, np.repeat(self._sl.reshape(1, -1), n_samp, axis=0))
        )
        self._nl_hist = np.concatenate(
            (self._nl_hist, np.repeat(self._nl.reshape(1, -1), n_samp, axis=0))
        )

        # Pack results in a DataFrame
        ics = pd.DataFrame(
            data=ics_array.T,
            index=[
                i / self._fs
                for i in range(self._n_samp_seen, self._n_samp_seen + n_samp)
            ],
            columns=[f"MU{i}" for i in range(ics_array.shape[0])],
        )

        self._n_samp_seen += n_samp

        return ics, spikes_t
