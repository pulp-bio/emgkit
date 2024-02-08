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
import pickle
import time

import numpy as np
import pandas as pd
import torch

from .. import spike_stats, utils
from .._base import Signal, signal_to_array
from ..preprocessing import PCAWhitening, WhiteningModel, ZCAWhitening, extend_signal
from . import _contrast_functions as cf


class ConvBSS:
    """Decompose EMG signals via convolutive blind source separation.

    Parameters
    ----------
    fs : float
        Sampling frequency of the signal.
    n_mu_target : int or str, default="same_ext"
        Number of target MUs to extract:
        - if set to the string "same_ext", it will be set to the number of extended observations;
        - otherwise, it will be set to the given number.
    f_ext_ms : float or str, default="same_n_mu"
        Extension factor for the signal (in ms):
        - if set to the string "same_n_mu", it will be set to n. of target MUs / n. of channels
        (if n_mu is "same_ext", extension will be disabled);
        - if set to the string "auto", it will be set to 1000 / n. of channels;
        - if set to the string "none", the signal won't be extended;
        - otherwise, it will be set to the given value.
    g_name : {"logcosh", "skewness", "gauss", "kurtosis", "rati"}, default="logcosh"
        Name of the contrast function.
    conv_th : float, default=1e-4
        Threshold for convergence.
    max_iter : int, default=200
        Maximum n. of iterations.
    sil_th : float, default=0.85
        Minimum silhouette threshold for considering a MU as valid.
    dr_th : float, default=5.0
        Minimum discharge rate (in spikes/s) for considering a MU as valid.
    cov_isi_th : float, default=0.5
        Maximum CoV-ISI for considering a MU as valid.
    device : device or str, default="cpu"
        Torch device.
    seed : int or None, default=None
        Seed for the internal PRNG.
    whiten_alg : {"zca", "pca"}, default="zca"
        Whitening algorithm.
    whiten_kw : dict or None, default=None
        Whitening arguments.
    bin_alg : {"kmeans", "otsu"}, default="kmeans"
        Binarization algorithm.
    ref_period_ms : float, default=20.0
        Refractory period for spike detection (in ms).
    dup_perc : float, default=0.3
        Minimum percentage of synchronized discharges for considering two MUs as duplicates.
    dup_tol_ms : float, default=0.5
        Tolerance (in ms) for considering two discharges as synchronized.

    Attributes
    ----------
    _fs : float
        Sampling frequency of the signal.
    _n_mu_target : int
        Number of target MUs to extract.
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
    _dr_th : float
        Minimum discharge rate (in spikes/s) for considering a MU as valid.
    _device : device
        Torch device.
    _prng : Generator
        Actual PRNG.
    _bin_alg : str
        Binarization algorithm.
    _ref_period : float
        Refractory period for spike detection.
    _dup_perc : float
        Minimum percentage of synchronized discharges for considering two MUs as duplicates.
    _dup_tol_ms : float
        Tolerance (in ms) for considering two discharges as synchronized.
    """

    def __init__(
        self,
        fs: float,
        n_mu_target: int | str = "same_ext",
        f_ext_ms: float | str = "same_n_mu",
        g_name: str = "logcosh",
        conv_th: float = 1e-4,
        max_iter: int = 100,
        sil_th: float = 0.85,
        cov_isi_th: float = 0.5,
        dr_th: float = 5.0,
        device: torch.device | str = "cpu",
        seed: int | None = None,
        whiten_alg: str = "zca",
        whiten_kw: dict | None = None,
        bin_alg: str = "kmeans",
        ref_period_ms: float = 20.0,
        dup_perc: float = 0.3,
        dup_tol_ms: float = 0.5,
    ):
        assert (isinstance(n_mu_target, int) and n_mu_target > 0) or (
            isinstance(n_mu_target, str) and n_mu_target == "same_ext"
        ), 'n_mu must be either a positive integer or "same_ext".'
        assert (isinstance(f_ext_ms, float) and f_ext_ms > 0) or (
            isinstance(f_ext_ms, str) and f_ext_ms in ("same_n_mu", "auto", "none")
        ), 'f_ext_ms must be either a positive float, "same_n_mu", "auto" or "none".'
        assert g_name in (
            "logcosh",
            "skewness",
            "gauss",
            "kurtosis",
            "rati",
        ), (
            'Contrast function can be either "logcosh", "skewness", "gauss", "kurtosis" or "rati": '
            f'the provided one was "{g_name}".'
        )
        assert conv_th > 0, "Convergence threshold must be positive."
        assert max_iter > 0, "The maximum n. of iterations must be positive."
        assert whiten_alg in (
            "zca",
            "pca",
        ), f'Whitening algorithm must be either "zca" or "pca": the provided one was {whiten_alg}'
        assert bin_alg in (
            "kmeans",
            "otsu",
        ), f'Binarization algorithm must be either "kmeans" or "otsu": the provided one was {bin_alg}'

        self._fs = fs

        self._device = torch.device(device) if isinstance(device, str) else device

        # Whitening model
        whiten_dict = {
            "zca": ZCAWhitening,
            "pca": PCAWhitening,
        }
        whiten_kw = {} if whiten_kw is None else whiten_kw
        whiten_kw["device"] = self._device
        self._whiten_model: WhiteningModel = whiten_dict[whiten_alg](**whiten_kw)

        # Map "same_ext" -> 0
        self._n_mu_target = 0 if n_mu_target == "same_ext" else n_mu_target

        # Map "same_n_mu" -> 0, "auto" -> -1 and "none" -> 1
        if f_ext_ms == "same_n_mu":
            self._f_ext = 0
        elif f_ext_ms == "auto":
            self._f_ext = -1
        elif f_ext_ms == "none":
            self._f_ext = 1
        else:
            self._f_ext = int(round(f_ext_ms / 1000 * fs))

        g_dict = {
            "logcosh": cf.logcosh,
            "skewness": cf.skewness,
            "gauss": cf.gauss,
            "kurtosis": cf.kurtosis,
            "rati": cf.rati,
        }
        self._g_func = g_dict[g_name]
        self._conv_th = conv_th
        self._max_iter = max_iter
        self._sil_th = sil_th
        self._cov_isi_th = cov_isi_th
        self._dr_th = dr_th
        self._prng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self._bin_alg = bin_alg
        self._ref_period = int(round(ref_period_ms / 1000 * fs))
        self._dup_perc = dup_perc
        self._dup_tol_ms = dup_tol_ms

        self._n_mu = 0

        self._sep_mtx: torch.Tensor = None  # type: ignore
        self._spike_ths: np.ndarray = None  # type: ignore

    @property
    def whiten_model(self) -> WhiteningModel:
        """WhiteningModel: Property for getting the whitening model."""
        return self._whiten_model

    @property
    def sep_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated separation matrix."""
        return self._sep_mtx

    @property
    def spike_ths(self) -> np.ndarray:
        """ndarray: Property for getting the estimated separation matrix."""
        return self._spike_ths

    @property
    def n_mu(self) -> int:
        """int: Property for getting the number of identified motor units."""
        return self._n_mu

    @property
    def f_ext(self) -> int:
        """int: Property for getting the extension factor."""
        return self._f_ext

    def save_to_file(self, filename: str) -> None:
        """Save instance to a .pkl file using pickle.

        Parameters
        ----------
        filename : str
            Path to the .pkl file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename: str) -> ConvBSS:
        """Load instance from a .pkl file using pickle.

        Parameters
        ----------
        filename : str
            Path to the .pkl file.

        Returns
        -------
        EMGSeparator
            Instance of EMGSeparator.
        """
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj

    def decompose_training(
        self, emg: Signal
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Train the decomposition model to decompose the given EMG signal into MUAPTs. If called multiple times,
        the model updates its internal parameters without forgetting the previous history.

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
        start = time.time()

        # Convert to array
        emg_array = signal_to_array(emg)
        n_ch = emg_array.shape[1]

        # Decide extension factor
        if self._f_ext < 0:  # "auto"
            # Apply heuristic
            self._f_ext = int(round(1000 / n_ch))
        elif self._f_ext == 0:  # "same_n_mu"
            # Check n_mu: if it's set to "same_ext", disable extension
            self._f_ext = (
                1
                if self._n_mu_target == 0
                else int(math.ceil(self._n_mu_target / n_ch))
            )

        # 1. Extension
        logging.info(f"Number of channels before extension: {n_ch}")
        emg_ext = extend_signal(emg_array, self._f_ext)
        n_samp, n_ch_ext = emg_ext.shape
        logging.info(f"Number of channels after extension: {n_ch_ext}")

        # 2. Whitening
        emg_white = self._whiten_model.whiten_training(emg_ext).T
        n_ch_w = emg_white.size(0)

        if self._n_mu_target == 0:
            self._n_mu_target = n_ch_w

        # 3. ICA
        sep_mtx = torch.zeros(
            self._n_mu_target,
            n_ch_w,
            dtype=emg_white.dtype,
            device=self._device,
        )
        spike_ths = np.zeros(shape=self._n_mu_target, dtype=emg_array.dtype)
        ics = torch.zeros(
            self._n_mu_target,
            n_samp,
            dtype=emg_white.dtype,
            device=self._device,
        )
        spikes_t = []
        sil_scores = []
        w_init = self._initialize_weights(emg_white)
        for i in range(self._n_mu_target):
            logging.info(f"----- IC {i + 1} -----")

            if i < self._n_mu:  # ICs from previous training
                w_init_i = self._sep_mtx[i]  # initialize with old separation vector
            else:  # ICs from current training
                w_init_i = w_init[i - self._n_mu]

            w_i = self._fast_ica_iter(emg_white, sep_mtx, w_i_init=w_init_i)
            # Solve sign uncertainty
            ic_i = w_i @ emg_white
            if (ic_i**3).mean() < 0:
                w_i *= -1

            # CoV-ISI improvement
            w_i, spikes_i, spike_th_i, sil = self._cov_isi_improvement(
                emg_white, w_i=w_i
            )

            # Save separation vector and spike/noise threshold
            sep_mtx[i] = w_i
            spike_ths[i] = spike_th_i
            # Save current IC, discharge times and SIL
            ics[i] = w_i @ emg_white
            spikes_t.append(spikes_i / self._fs)
            sil_scores.append(sil)

        # 4. Post-processing
        # 4.1. SIL, CoV-ISI and DR thresholding
        idx_to_keep = []
        cov_isi_scores = []
        for i in range(self._n_mu_target):
            # Check SIL
            sil = sil_scores[i]
            if np.isnan(sil) or sil <= self._sil_th:
                logging.info(
                    f"{i}-th IC: SIL below threshold (SIL = {sil:.3f} <= {self._sil_th:.3f}) -> skipped."
                )
                continue

            # Check CoV-ISI
            cov_isi = spike_stats.cov_isi(spikes_t[i])
            if np.isnan(cov_isi) or cov_isi >= self._cov_isi_th:
                logging.info(
                    f"{i}-th IC: CoV-ISI above threshold (CoV-ISI = {cov_isi:.2%} >= {self._cov_isi_th:.2%})"
                    f" -> skipped."
                )
                continue

            # Check discharge rate
            avg_dr = spike_stats.instantaneous_discharge_rate(spikes_t[i]).mean()
            if avg_dr <= self._dr_th:
                logging.info(
                    f"{i}-th IC: discharge rate below threshold (DR = {avg_dr:.3f} <= {self._dr_th:.3f})"
                    f" -> skipped."
                )
                continue

            logging.info(
                f"{i}-th IC: SIL = {sil:.3f}, CoV-ISI = {cov_isi:.2%} -> accepted."
            )
            cov_isi_scores.append(cov_isi)
            idx_to_keep.append(i)
        sep_mtx = sep_mtx[idx_to_keep]
        spike_ths = spike_ths[idx_to_keep]
        ics = ics[idx_to_keep]
        spikes_t = {f"MU{i}": spikes_t[idx] for i, idx in enumerate(idx_to_keep)}
        cov_isi_scores = {i: cov_isi for i, cov_isi in enumerate(cov_isi_scores)}

        logging.info(f"Extracted {len(spikes_t)} MUs after post-processing.")

        # 4.2. Replicas removal
        logging.info("Looking for delayed replicas...")
        ics_bin = utils.sparse_to_dense(spikes_t, n_samp / self._fs, self._fs)
        duplicate_mus = utils.find_replicas(
            ics_bin, fs=self._fs, tol_ms=self._dup_tol_ms, min_perc=self._dup_perc
        )
        idx_to_keep = list(range(len(spikes_t)))
        for main_mu, dup_mus in duplicate_mus.items():
            # Unify duplicate MUs
            dup_mus = [main_mu] + dup_mus
            dup_str = ", ".join([f"{mu}" for mu in dup_mus])
            logging.info(f"Found group of duplicate MUs: {dup_str}.")

            # Keep only the MU with the lowest CoV-ISI
            cov_isi_dup = {k: v for k, v in cov_isi_scores.items() if k in dup_mus}
            mu_keep = min(cov_isi_dup, key=lambda k: cov_isi_dup[k])
            logging.info(
                f"Keeping MU {mu_keep} (CoV-ISI = {cov_isi_dup[mu_keep]:.2%})."
            )

            # Mark duplicates
            dup_mus.remove(mu_keep)
            idx_to_keep = [i for i in idx_to_keep if i not in dup_mus]
        self._sep_mtx = sep_mtx[idx_to_keep]
        self._spike_ths = spike_ths[idx_to_keep]
        ics = ics[idx_to_keep]
        spikes_t = {f"MU{i}": spikes_t[f"MU{k}"] for i, k in enumerate(idx_to_keep)}
        self._n_mu = len(spikes_t)

        logging.info(f"Extracted {self._n_mu} MUs after replicas removal.")

        # Pack results in a DataFrame
        ics = pd.DataFrame(
            data=ics.T.cpu().numpy(),
            index=[i / self._fs for i in range(n_samp)],
            columns=[f"MU{i}" for i in range(self._n_mu)],
        )

        elapsed = int(round(time.time() - start))
        mins, secs = divmod(elapsed, 60)
        logging.info(f"Decomposition performed in {mins:02d}min {secs:02d}s.")

        return ics, spikes_t

    def decompose_inference(
        self, emg: Signal
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Decompose the given EMG signal into MUAPTs using the frozen decomposition model.

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
        is_fit = self._sep_mtx is not None and self._spike_ths is not None
        assert is_fit, "Fit the model first."

        # 1. Extension
        emg_ext = extend_signal(emg, self._f_ext)
        n_samp = emg_ext.shape[0]
        emg_ext = torch.from_numpy(emg_ext).to(self._device)

        # 2. Whitening + ICA
        ics = self._whiten_model.whiten_inference(emg_ext) @ self._sep_mtx.T

        # 3. Spike extraction
        spikes_t = {}
        for i in range(self._n_mu):
            spikes_i = utils.detect_spikes(
                ics[:, i],
                ref_period=self._ref_period,
                bin_alg=self._bin_alg,
                threshold=self._spike_ths[i].item(),
                seed=self._prng,
            )[0]
            spikes_t[f"MU{i}"] = spikes_i / self._fs

        # Pack results in a DataFrame
        ics = pd.DataFrame(
            data=ics.cpu().numpy(),
            index=[i / self._fs for i in range(n_samp)],
            columns=[f"MU{i}" for i in range(self._n_mu)],
        )

        return ics, spikes_t

    def _initialize_weights(self, emg_white: torch.Tensor) -> torch.Tensor:
        """Initialize separation vectors."""
        gamma = emg_white.sum(dim=0) ** 2  # activation index
        w_init_idx = torch.topk(gamma, k=self._n_mu_target - self._n_mu).indices
        return emg_white[:, w_init_idx].T

    def _fast_ica_iter(
        self, x_w: torch.Tensor, sep_mtx: torch.Tensor, w_i_init: torch.Tensor
    ) -> torch.Tensor:
        """FastICA iteration."""
        w_i = w_i_init
        w_i /= torch.linalg.norm(w_i)
        decorr_mtx = sep_mtx.T @ sep_mtx

        iter_idx = 1
        while iter_idx <= self._max_iter:
            g_res = self._g_func(w_i @ x_w)
            w_i_new = (x_w * g_res.g1_u).mean(dim=1) - g_res.g2_u.mean() * w_i
            w_i_new -= w_i_new @ decorr_mtx  # Gram-Schmidt decorrelation
            w_i_new /= torch.linalg.norm(w_i_new)

            distance = 1 - abs((w_i_new @ w_i).item())
            w_i = w_i_new
            if distance < self._conv_th:
                logging.info(
                    f"FastICA converged after {iter_idx} iterations, the distance is: {distance:.3e}."
                )
                break
            iter_idx += 1

        return w_i

    def _cov_isi_improvement(
        self,
        emg_white: torch.Tensor,
        w_i: torch.Tensor,
    ) -> tuple[torch.Tensor, np.ndarray, float, float]:
        """CoV-ISI improvement iteration."""

        ic_i = w_i @ emg_white
        spikes, spike_th, sil = utils.detect_spikes(
            ic_i,
            ref_period=self._ref_period,
            bin_alg=self._bin_alg,
            compute_sil=True,
            seed=self._prng,
        )
        cov_isi = spike_stats.cov_isi(spikes / self._fs)
        iter_idx = 0
        if math.isnan(cov_isi):
            logging.info("Spike detection failed.")
            return w_i, spikes, np.nan, np.nan

        while True:
            w_i_new = emg_white[:, spikes].sum(dim=1)
            w_i_new /= torch.linalg.norm(w_i_new)

            ic_i = w_i_new @ emg_white
            spikes_new, spike_th_new, sil_new = utils.detect_spikes(
                ic_i,
                ref_period=self._ref_period,
                bin_alg=self._bin_alg,
                compute_sil=True,
                seed=self._prng,
            )
            cov_isi_new = spike_stats.cov_isi(spikes_new / self._fs)
            iter_idx += 1

            if math.isnan(cov_isi_new):
                logging.info(f"Spike detection failed after {iter_idx} steps.")
                break
            if cov_isi_new >= cov_isi:
                logging.info(
                    f"CoV-ISI increased from {cov_isi:.2%} to {cov_isi_new:.2%} "
                    f"after {iter_idx} steps."
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
