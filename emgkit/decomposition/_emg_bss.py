"""Class implementing the convolutive blind source separation algorithm
for EMG decomposition (https://doi.org/10.1088/1741-2560/13/2/026027).


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

import logging
import math
import time

import numpy as np
import pandas as pd
import torch

from .. import preprocessing, spike_stats, utils
from .._base import Signal, signal_to_array
from ..ica import contrast_functions as cf


class EMGBSS:
    """Decompose EMG signals via convolutive blind source separation.

    Parameters
    ----------
    fs : float
        Sampling frequency of the signal.
    f_ext_ms : float, default=-1
        Extension factor for the signal (in ms):
        - if zero, the signal won't be extended;
        - if negative, it will be set to 1000 / n. of channels.
    n_mu_target : int, default=-1
        Number of target MUs to extract (if zero or negative, it will be set to the number of extended observations).
    ref_period_ms : float, default=20.0
        Refractory period for spike detection (in ms).
    g_name : {"logcosh", "gauss", "kurtosis", "skewness", "rati"}, default="logcosh"
        Name of the contrast function.
    conv_th : float, default=1e-4
        Threshold for convergence.
    max_iter : int, default=200
        Maximum n. of iterations.
    sil_th : float, default=0.85
        Minimum silhouette threshold for considering a MU as valid.
    cov_isi_th : float, default=0.5
        Maximum CoV-ISI for considering a MU as valid.
    dup_perc : float, default=0.3
        Minimum percentage of synchronized discharges for considering two MUs as duplicates.
    dup_tol_ms : float, default=0.5
        Tolerance (in ms) for considering two discharges as synchronized.
    seed : int or None, default=None
        Seed for the internal PRNG.
    device : device or str or None, default=None
        Torch device.

    Attributes
    ----------
    _fs : float
        Sampling frequency of the signal.
    _n_mu_target : int
        Number of target MUs to extract.
    _ref_period : float
        Refractory period for spike detection.
    _g_func : ContrastFunction
        Contrast function.
    _conv_th : float
        Threshold for convergence.
    _max_iter : int
        Maximum n. of iterations.
    _sil_th : float
        Minimum silhouette threshold for considering a MU as valid.
    _cov_isi_th : float
        Maximum CoV-ISI for considering a MU as valid.
    _dup_perc : float
        Minimum percentage of synchronized discharges for considering two MUs as duplicates.
    _dup_tol_ms : float
        Tolerance (in ms) for considering two discharges as synchronized.
    _prng : Generator
        Actual PRNG.
    _device : device or None
        Torch device.
    """

    def __init__(
        self,
        fs: float,
        f_ext_ms: float = -1,
        n_mu_target: int = -1,
        ref_period_ms: float = 20.0,
        g_name: str = "logcosh",
        conv_th: float = 1e-4,
        max_iter: int = 100,
        sil_th: float = 0.85,
        cov_isi_th: float = 0.5,
        dup_perc: float = 0.3,
        dup_tol_ms: float = 0.5,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ):
        assert g_name in (
            "logcosh",
            "gauss",
            "kurtosis",
            "skewness",
            "rati",
        ), (
            'Contrast function can be either "logcosh", "gauss", "kurtosis", "skewness" or "rati": '
            f'the provided one was "{g_name}".'
        )
        assert conv_th > 0, "Convergence threshold must be positive."
        assert max_iter > 0, "The maximum n. of iterations must be positive."

        self._fs: float = fs
        self._n_mu_target: int = n_mu_target
        self._ref_period: int = int(round(ref_period_ms / 1000 * fs))
        g_dict = {
            "logcosh": cf.logcosh,
            "gauss": cf.gauss,
            "kurtosis": cf.kurtosis,
            "skewness": cf.skewness,
            "rati": cf.rati,
        }
        self._g_func: cf.ContrastFunction = g_dict[g_name]
        self._conv_th: float = conv_th
        self._max_iter: int = max_iter
        self._sil_th: float = sil_th
        self._cov_isi_th: float = cov_isi_th
        self._dup_perc: float = dup_perc
        self._dup_tol_ms: float = dup_tol_ms
        self._prng: np.random.Generator = np.random.default_rng(seed)
        self._device: torch.device | None = (
            torch.device(device) if isinstance(device, str) else device
        )

        if f_ext_ms == 0:  # disable extension
            self._f_ext: int = 1
        elif f_ext_ms < 0:  # apply heuristic later
            self._f_ext: int = int(f_ext_ms)
        else:  # convert from ms to samples
            self._f_ext: int = int(round(f_ext_ms / 1000 * fs))

        if seed is not None:
            torch.manual_seed(seed)

        self._mean_vec: torch.Tensor | None = None
        self._white_mtx: torch.Tensor | None = None
        self._sep_mtx: torch.Tensor | None = None
        self._spike_ths: np.ndarray | None = None

    @property
    def mean_vec(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated mean vector."""
        return self._mean_vec

    @property
    def white_mtx(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated whitening matrix."""
        return self._white_mtx

    @property
    def sep_mtx(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated separation matrix."""
        return self._sep_mtx

    @property
    def spike_ths(self) -> np.ndarray | None:
        """ndarray or None: Property for getting the estimated separation matrix."""
        return self._spike_ths

    @property
    def n_mu(self) -> int:
        """int: Property for getting the number of identified motor units."""
        return self._sep_mtx.size(dim=0) if self._sep_mtx is not None else -1

    @property
    def f_ext(self) -> int:
        """int: Property for getting the extension factor."""
        return self._f_ext

    def fit(self, emg: Signal) -> EMGBSS:
        """Fit the decomposition model on the given data.

        Parameters
        ----------
        emg : Signal
            EMG signal with shape (n_samples, n_channels).

        Returns
        -------
        EMGBSS
            The fitted instance of the decomposition model.
        """
        # Fit the model and return self
        self._fit_transform(emg)
        return self

    def fit_transform(self, emg: Signal) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Fit the decomposition model on the given data and compute the estimated MUAPTs.

        Parameters
        ----------
        emg : Signal
            EMG signal with shape (n_samples, n_channels).

        Returns
        -------
        DataFrame
            A DataFrame with shape (n_samples, n_mu) containing the components estimated by ICA.
        dict of {str : ndarray}
            Dictionary containing the discharge times for each MU.
        """
        # Fit the model and return result
        return self._fit_transform(emg)

    def transform(self, emg: Signal) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Compute the estimated MUAPTs using the fitted decomposition model.

        Parameters
        ----------
        emg : Signal
            EMG signal with shape (n_samples, n_channels).

        Returns
        -------
        DataFrame
            A DataFrame with shape (n_samples, n_mu) containing the components estimated by ICA.
        dict of {str : ndarray}
            Dictionary containing the discharge times for each MU.
        """
        assert (
            self._mean_vec is not None
            and self._sep_mtx is not None
            and self._spike_ths is not None
        ), "Mean vector, separation matrix or spike-noise thresholds are null, fit the model first."

        # 1. Extension
        emg_ext = preprocessing.extend_signal(emg, self._f_ext)
        n_samp = emg_ext.shape[0]
        emg_ext = torch.from_numpy(emg_ext).to(self._device).T

        # 2. Whitening + ICA
        ics = self._sep_mtx @ self._white_mtx @ (emg_ext - self._mean_vec)

        # 3. Spike extraction
        n_mu = ics.shape[0]
        spikes_t = {}
        for i in range(n_mu):
            spikes_i = utils.detect_spikes(
                ics[i],
                ref_period=self._ref_period,
                bin_alg="kmeans",
                threshold=self._spike_ths[i].item(),
                seed=self._prng,
            )[0]
            spikes_t[f"MU{i}"] = spikes_i / self._fs

        # Pack results in a DataFrame
        ics = pd.DataFrame(
            data=ics.cpu().T,
            index=[i / self._fs for i in range(n_samp)],
            columns=[f"MU{i}" for i in range(n_mu)],
        )

        return ics, spikes_t

    def _fit_transform(self, emg: Signal) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Helper method for fit and fit_transform."""
        start = time.time()

        # Convert to array
        emg_array = signal_to_array(emg)
        n_ch = emg_array.shape[1]

        # Apply heuristic
        if self._f_ext < 0:
            self._f_ext = int(round(1000 / n_ch))

        # 1. Extension
        logging.info(f"Number of channels before extension: {n_ch}")
        emg_ext = preprocessing.extend_signal(emg_array, self._f_ext)
        n_samp = emg_ext.shape[0]
        logging.info(f"Number of channels after extension: {emg_ext.shape[1]}")

        # 2. Whitening
        emg_white, self._mean_vec, self._white_mtx = preprocessing.zca_whitening(
            emg_ext, device=self._device
        )
        emg_white = emg_white.T

        if self._n_mu_target <= 0:
            self._n_mu_target = emg_white.size(0)

        # 3. ICA
        self._sep_mtx = torch.zeros(
            0, emg_white.size(0), dtype=emg_white.dtype, device=self._device
        )
        self._spike_ths = np.zeros(shape=0, dtype=emg_array.dtype)
        w_init = self._initialize_weights(emg_white)
        ics = torch.zeros(0, n_samp, dtype=emg_white.dtype, device=self._device)
        spikes_t = {}
        sil_scores = []
        idx = 0
        for i in range(self._n_mu_target):
            logging.info(f"----- IC {i + 1} -----")

            w_i, converged = self._fast_ica_iter(emg_white, w_i_init=w_init[i])
            if not converged:
                logging.info("FastICA didn't converge, reinitializing...")
                continue

            # Solve sign uncertainty
            ic_i = w_i @ emg_white
            if (ic_i**3).mean() < 0:
                w_i *= -1

            # CoV-ISI improvement
            w_i, spikes_i, spike_th_i, sil = self._cov_isi_improvement(
                emg_white, w_i=w_i
            )
            spikes_t_i = spikes_i / self._fs
            if w_i is None:
                logging.info("IC improvement iteration failed, skipping IC.")
                continue

            if sil <= self._sil_th:
                logging.info(
                    f"SIL below threshold (SIL = {sil:.3f} <= {self._sil_th:.3f}), skipping IC."
                )
                continue

            cov_isi = spike_stats.cov_isi(spikes_t_i)
            if cov_isi >= self._cov_isi_th:
                logging.info(
                    f"CoV-ISI above threshold (CoV-ISI = {cov_isi:.3f} >= {self._cov_isi_th:.3f}), skipping IC."
                )
                continue

            # Save separation vector and spike/noise threshold
            self._sep_mtx = torch.vstack((self._sep_mtx, w_i))
            self._spike_ths = np.append(self._spike_ths, spike_th_i)
            logging.info(f"SIL = {sil:.3f}")
            logging.info(f"CoV-ISI = {cov_isi:.2%}")
            logging.info(f"-> MU accepted (n. of MUs: {self._sep_mtx.shape[0]}).")

            # Save current IC, discharge times and SIL
            ic_i = w_i @ emg_white
            ics = torch.vstack((ics, ic_i))
            spikes_t[f"MU{idx}"] = spikes_t_i
            sil_scores.append((idx, sil))
            idx += 1

        logging.info(f"Extracted {len(spikes_t)} MUs before replicas removal.")

        # 5. Duplicates removal
        logging.info("Looking for delayed replicas...")
        ics_bin = utils.sparse_to_dense(spikes_t, n_samp / self._fs, self._fs)
        duplicate_mus = utils.find_replicas(
            ics_bin, fs=self._fs, tol_ms=self._dup_tol_ms, min_perc=self._dup_perc
        )
        mus_to_remove = []
        for main_mu, dup_mus in duplicate_mus.items():
            # Unify duplicate MUs
            dup_mus = [main_mu] + dup_mus
            dup_str = ", ".join([f"{mu}" for mu in dup_mus])
            logging.info(f"Found group of duplicate MUs: {dup_str}.")

            # Keep only the MU with the highest SIL
            sil_scores_dup = list(filter(lambda t: t[0] in dup_mus, sil_scores))
            mu_keep = max(sil_scores_dup, key=lambda t: t[1])
            logging.info(f"Keeping MU {mu_keep[0]} (SIL = {mu_keep[1]:.3f}).")

            # Mark duplicates
            dup_mus.remove(mu_keep[0])
            mus_to_remove.extend(dup_mus)
        mus_to_remove = set(mus_to_remove)
        # Remove duplicates
        mus_to_keep = {i for i in range(len(spikes_t))}
        mus_to_keep -= mus_to_remove
        mus_to_keep = list(mus_to_keep)
        self._sep_mtx = self._sep_mtx[mus_to_keep]
        self._spike_ths = self._spike_ths[mus_to_keep]
        ics = ics[mus_to_keep]
        spikes_t = {f"MU{i}": spikes_t[f"MU{k}"] for i, k in enumerate(mus_to_keep)}
        n_mu = len(spikes_t)

        logging.info(f"Extracted {n_mu} MUs after replicas removal.")

        # Pack results in a DataFrame
        ics = pd.DataFrame(
            data=ics.cpu().T,
            index=[i / self._fs for i in range(n_samp)],
            columns=[f"MU{i}" for i in range(n_mu)],
        )

        elapsed = int(round(time.time() - start))
        hours, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        logging.info(
            f"Decomposition performed in {hours:d}h {mins:02d}min {secs:02d}s."
        )

        return ics, spikes_t

    def _initialize_weights(self, emg_white: torch.Tensor) -> torch.Tensor:
        """Initialize separation vectors."""

        gamma = (emg_white**2).sum(dim=0)  # activation index
        w_init_idx = torch.topk(gamma, k=self._n_mu_target).indices
        return emg_white[:, w_init_idx].T

    def _fast_ica_iter(
        self, x_w: torch.Tensor, w_i_init: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """FastICA iteration."""
        w_i = w_i_init
        w_i /= torch.linalg.norm(w_i)

        iter_idx = 1
        converged = False
        while iter_idx <= self._max_iter:
            g_res = self._g_func(w_i @ x_w)
            w_i_new = (x_w * g_res.g1_u).mean(dim=1) - g_res.g2_u.mean() * w_i
            w_i_new -= (
                w_i_new @ self._sep_mtx.T @ self._sep_mtx
            )  # Gram-Schmidt decorrelation
            w_i_new /= torch.linalg.norm(w_i_new)

            distance = 1 - abs((w_i_new @ w_i).item())
            logging.info(f"FastICA iteration {iter_idx}: {distance:.3e}.")

            w_i = w_i_new

            if distance < self._conv_th:
                converged = True
                logging.info(
                    f"FastICA converged after {iter_idx} iterations, the distance is: {distance:.3e}."
                )
                break

            iter_idx += 1

        return w_i, converged

    def _cov_isi_improvement(
        self,
        emg_white: torch.Tensor,
        w_i: torch.Tensor,
    ) -> tuple[torch.Tensor | None, np.ndarray, float, float]:
        """CoV-ISI improvement iteration."""

        ic_i = w_i @ emg_white
        spikes, spike_th, sil = utils.detect_spikes(
            ic_i,
            ref_period=self._ref_period,
            bin_alg="kmeans",
            compute_sil=True,
            seed=self._prng,
        )
        cov_isi = spike_stats.cov_isi(spikes / self._fs)
        iter_idx = 0
        if math.isnan(cov_isi):
            logging.info("Spike detection failed, aborting.")
            return None, spikes, np.nan, np.nan

        while True:
            w_i_new = emg_white[:, spikes].mean(dim=1)
            w_i_new /= torch.linalg.norm(w_i_new)

            ic_i = w_i_new @ emg_white
            spikes_new, spike_th_new, sil_new = utils.detect_spikes(
                ic_i,
                ref_period=self._ref_period,
                bin_alg="kmeans",
                compute_sil=True,
                seed=self._prng,
            )
            cov_isi_new = spike_stats.cov_isi(spikes_new / self._fs)
            iter_idx += 1

            if math.isnan(cov_isi_new):
                logging.info(
                    f"Spike detection failed after {iter_idx} steps, aborting."
                )
                break
            if cov_isi_new >= cov_isi:
                logging.info(
                    f"CoV-ISI increased from {cov_isi:.2%} to {cov_isi_new:.2%} "
                    f"after {iter_idx} steps, aborting."
                )
                break
            logging.info(
                f"CoV-ISI decreased from {cov_isi:.2%} to {cov_isi_new:.2%} "
                f"after {iter_idx} steps."
            )
            w_i = w_i_new
            cov_isi = cov_isi_new
            spikes = spikes_new
            spike_th = spike_th_new
            sil = sil_new

        return w_i, spikes, spike_th, sil
