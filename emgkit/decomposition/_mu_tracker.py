"""
Class implementing the MU tracking algorithm.


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
    """
    Class implementing the MU tracking algorithm based on
    the Amari et al.'s algorithm with natural gradient and on Pan-Tompkins.

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
    use_adam: bool, default=False
        Whether to use Adam.
    device : device or str, default="cpu"
        Torch device.
    momentum : float, default = 0.6
        Momentum.
    beta1 : float = 0.9
        Decay rate of the first-moment estimates (relevant only if the optimizer is "adam").
    beta2 : float = 0.999
        Decay rate of the second-moment estimates (relevant only if the optimizer is "adam").

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
    _lr : float
        Learning rate.
    _use_adam: bool
        Whether to use Adam.
    _momentum : float
        Momentum.
    _beta1 : float
        Decay rate of the first-moment estimates (relevant only if the optimizer is "adam").
    _beta2 : float
        Decay rate of the second-moment estimates (relevant only if the optimizer is "adam").
    _white_m : float
        1st moment vector for whitening optimization.
    _white_v : float
        2nd moment vector for whitening optimization.
    _sep_m : float
        1st moment vector for separation optimization.
    _sep_v : float
        2nd moment vector for separation optimization.
    _sl : list of float
        Running estimate of spike level for each MU in the BSS output before integration.
    _nl : list of float
        Running estimate of noise level for each MU in the BSS output before integration.
    _n_samp_seen : int
        Number of samples seen.
    _t : int
        Number of iterations.
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
        use_adam: bool = False,
        device: torch.device | str = "cpu",
        momentum: float = 0.6,
        beta1: float = 0.9,
        beta2: float = 0.999,
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
        self._lr = learning_rate
        self._use_adam = use_adam
        self._momentum = momentum
        self._beta1 = beta1
        self._beta2 = beta2
        self._white_m = 0
        self._white_v = 0
        self._sep_m = 0
        self._sep_v = 0

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
        self._t = 1

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
        """
        Process a window and adapt internal parameters.

        Parameters
        ----------
        emg : Signal
            A window with shape (n_samples, n_channels).

        Returns
        -------
        DataFrame
            A DataFrame with shape (n_samples, n_mu) containing the components estimated by ICA.
        dict of str: ndarray
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
        grad_w = (
            self._white_mtx - emg_white @ emg_white.T / (n_samp - 1) @ self._white_mtx
        )
        if self._use_adam:  # Adam
            self._white_m = self._beta1 * self._white_m + (1 - self._beta1) * grad_w
            self._white_v = self._beta2 * self._white_v + (1 - self._beta2) * grad_w**2
            m_debias = self._white_m / (1 - self._beta1**self._t)
            v_debias = self._white_v / (1 - self._beta2**self._t)
            self._white_mtx += self._lr * m_debias / (v_debias.sqrt() + 1e-8)
        elif self._momentum != 0:  # momentum
            self._white_m = self._momentum * self._white_m + self._lr * grad_w
            self._white_mtx += self._white_m
        else:  # standard gradient update
            self._white_mtx += self._lr * grad_w
        emg_white = self._white_mtx @ emg_tensor

        # On-line BSS
        ics_tensor = self._sep_mtx @ emg_white
        g = -2 * torch.tanh(ics_tensor)
        grad_w = self._sep_mtx + g @ ics_tensor.T / (n_samp - 1) @ self._sep_mtx
        if self._use_adam:  # Adam
            self._sep_m = self._beta1 * self._sep_m + (1 - self._beta1) * grad_w
            self._sep_v = self._beta2 * self._sep_v + (1 - self._beta2) * grad_w**2
            m_debias = self._sep_m / (1 - self._beta1**self._t)
            v_debias = self._sep_v / (1 - self._beta2**self._t)
            self._sep_mtx += self._lr * m_debias / (v_debias.sqrt() + 1e-8)
        elif self._momentum != 0:  # momentum
            self._sep_m = self._momentum * self._sep_m + self._lr * grad_w
            self._sep_mtx += self._sep_m
        else:  # standard gradient update
            self._sep_mtx += self._lr * grad_w
        ics_tensor = self._sep_mtx @ emg_white

        # Solve sign uncertainty
        for i in range(ics_tensor.size(0)):
            if (ics_tensor[i] ** 3).mean() < 0:
                ics_tensor[i] *= -1
                self._sep_mtx[i] *= -1

        # Spike detection
        ics_array = ics_tensor.cpu().numpy()
        spikes_t = {}
        for i, ic_i in enumerate(ics_array):  # iterate over MUs
            spikes_t[f"MU{i}"] = np.asarray([], dtype=np.float32)

            # Detect peaks, and compare each one against signal and noise estimates
            peaks, _ = signal.find_peaks(
                ic_i, height=0, distance=int(round(20e-3 * self._fs))
            )
            for peak_idx in peaks:
                peak_val = ic_i[peak_idx]

                th = self._nl[i] + 0.5 * (self._sl[i] - self._nl[i])
                if peak_val < th:
                    self._nl[i] = 0.125 * peak_val + 0.875 * self._nl[i]
                    continue

                self._sl[i] = 0.125 * peak_val + 0.875 * self._sl[i]

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
        self._t += 1

        return ics, spikes_t
