"""
Class implementing the 2CFastICA algorithm
for EMG decomposition (https://doi.org/10.1109/TNSRE.2024.3398822).


Copyright 2025 Mattia Orlandi

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
from scipy import signal

from .. import preprocessing, spike_stats, utils
from .._base import Signal, signal_to_array
from . import _contrast_functions as cf


def _null_space(x: torch.Tensor, rcond: float | None = None) -> torch.Tensor:
    """Return an orthonormal basis for null(x) as columns (n, k) on same device."""
    m, n = x.shape
    _, s, vt = torch.linalg.svd(x, full_matrices=True)
    v = vt.T

    if rcond is None:
        rcond = torch.finfo(s.dtype).eps * max(m, n)
    tol = (
        s.max() * rcond
        if s.numel()
        else torch.tensor(0.0, dtype=s.dtype, device=s.device)
    )
    rank = int((s > tol).sum().item())

    # Nullspace basis consists of the last n-rank columns of V
    k = v[:, rank:]  # (n, n-rank)
    return k


class TwoCFastICA:
    """
    Decompose EMG signals via convolutive blind source separation.

    Parameters
    ----------
    fs : float
        Sampling frequency of the signal.
    n_mu_target : int or str, default="same_ext"
        Number of target MUs to extract:
        - if set to the string "same_ext", it will be set to the number of extended observations;
        - otherwise, it will be set to the given number.
    f_ext : int or str, default="same_n_mu"
        Extension factor for the signal:
        - if set to the string "same_n_mu", it will be set to n. of target MUs / n. of channels
        (if n_mu is "same_ext", extension will be disabled);
        - if set to the string "auto", it will be set to 1000 / n. of channels;
        - otherwise, it will be set to the given value.
    g_name : {"logcosh", "skewness", "gauss", "kurtosis", "rati"}, default="logcosh"
        Name of the contrast function.
    conv_th : float, default=1e-4
        Threshold for convergence.
    max_iter : int, default=200
        Maximum n. of iterations.
    sil_th : float, default=0.6
        Minimum silhouette threshold for considering a MU as valid.
    cov_isi_th : float, default=1.0
        Maximum CoV-ISI for considering a MU as valid.
    cov_isi_rest: bool, default=False
        Whether rest periods are admitted during CoV-ISI calculation.
    cov_amp_th : float, default=1.0
        Maximum CoV-Amp for considering a MU as valid.
    min_dr : float, default=5.0
        Minimum discharge rate (in spikes/s) for considering a MU as valid.
    max_dr : float, default=50.0
        Maximum discharge rate (in spikes/s) for considering a MU as valid.
    device : device or str, default="cpu"
        Torch device.
    seed : int or None, default=None
        Seed for the internal PRNG.
    whiten_alg : {"zca", "pca"}, default="zca"
        Whitening algorithm.
    square_ics : bool, default=True
        Whether to square the ICs before looking for spikes, or apply skewness correction.
    kwargs : dict
        Keyword arguments (for the PCA whitening procedure).

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
    _cov_isi_rest: bool
        Whether rest periods are admitted during CoV-ISI calculation.
    _cov_amp_th : float
        Maximum CoV-Amp for considering a MU as valid.
    _min_dr : float
        Minimum discharge rate (in spikes/s) for considering a MU as valid.
    _max_dr : float
        Maximum discharge rate (in spikes/s) for considering a MU as valid.
    _device : device
        Torch device.
    _prng : Generator
        Actual PRNG.
    _square_ics : bool
        Whether to square the ICs before looking for spikes, or apply skewness correction.
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
        f_ext: float | str = "same_n_mu",
        g_name: str = "logcosh",
        conv_th: float = 1e-4,
        max_iter: int = 100,
        sil_th: float = 0.6,
        cov_isi_th: float = 1.0,
        cov_isi_rest: bool = False,
        cov_amp_th: float = 1.0,
        min_dr: float = 5.0,
        max_dr: float = 50.0,
        device: torch.device | str = "cpu",
        seed: int | None = None,
        whiten_alg: str = "zca",
        square_ics: bool = True,
        **kwargs,
    ):
        assert (isinstance(n_mu_target, int) and n_mu_target > 0) or (
            isinstance(n_mu_target, str) and n_mu_target == "same_ext"
        ), 'n_mu must be either a positive integer or "same_ext".'
        assert (isinstance(f_ext, int) and f_ext > 0) or (
            isinstance(f_ext, str) and f_ext in ("same_n_mu", "auto")
        ), 'f_ext must be either a positive integer, "same_n_mu", "auto".'
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

        self._fs = fs

        self._device = torch.device(device) if isinstance(device, str) else device

        # Whitening model
        whiten_dict = {
            "zca": preprocessing.ZCAWhitening,
            "pca": preprocessing.PCAWhitening,
        }
        self._whiten_model: preprocessing.WhiteningModel = whiten_dict[whiten_alg](
            **kwargs, device=self._device
        )

        # Map "same_ext" -> 0
        self._n_mu_target = 0 if n_mu_target == "same_ext" else n_mu_target

        # Map "same_n_mu" -> 0 and "auto" -> -1
        if f_ext == "same_n_mu":
            self._f_ext = 0
        elif f_ext == "auto":
            self._f_ext = -1
        else:
            self._f_ext = f_ext

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
        self._cov_isi_rest = cov_isi_rest
        self._cov_amp_th = cov_amp_th
        self._min_dr = min_dr
        self._max_dr = max_dr
        self._prng = np.random.default_rng(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self._square_ics = square_ics
        self._ref_period = int(round(20e-3 * fs))  # 20ms

        self._n_mu = 0
        self._sep_mtx: torch.Tensor | None = None
        self._spike_ths: np.ndarray | None = None

    @property
    def whiten_model(self) -> preprocessing.WhiteningModel:
        """WhiteningModel: Property for getting the whitening model."""
        return self._whiten_model

    @property
    def sep_mtx(self) -> torch.Tensor:
        """
        Tensor: Property for getting the estimated separation matrix.

        Raises
        ------
        AttributeError
            If the decomposition model is not trained.
        """
        if self._sep_mtx is None:
            raise AttributeError(
                'The "sep_mtx" field is not available: call "decompose_training(x)" first.'
            )
        return self._sep_mtx

    @property
    def spike_ths(self) -> np.ndarray:
        """
        ndarray: Property for getting the estimated spike/noise thresholds.

        Raises
        ------
        AttributeError
            If the decomposition model is not trained.
        """
        if self._spike_ths is None:
            raise AttributeError(
                'The "spike_ths" field is not available: call "decompose_training(x)" first.'
            )
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
        """
        Save instance to a .pkl file using pickle.

        Parameters
        ----------
        filename : str
            Path to the .pkl file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename: str) -> TwoCFastICA:
        """
        Load instance from a .pkl file using pickle.

        Parameters
        ----------
        filename : str
            Path to the .pkl file.

        Returns
        -------
        TwoCFastICA
            Instance of TwoCFastICA.
        """
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj

    def decompose_training(
        self, emg: Signal
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """
        Train the decomposition model to decompose the given EMG signal into MUAPTs. If called multiple times,
        the model updates its internal parameters without forgetting the previous history.

        Parameters
        ----------
        emg : Signal
            EMG signal with shape (n_samples, n_channels).

        Returns
        -------
        DataFrame
            A DataFrame with shape (n_samples, n_mu) containing the components estimated by ICA.
        dict of str: ndarray
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
        emg_ext = preprocessing.extend_signal(emg_array, self._f_ext)
        n_samp, n_ch_ext = emg_ext.shape
        logging.info(f"Number of channels after extension: {n_ch_ext}")

        # 2. Whitening
        emg_white = self._whiten_model.whiten_training(emg_ext).T
        n_ch_w = emg_white.size(0)

        if self._n_mu_target == 0:
            self._n_mu_target = n_ch_w

        # 3. ICA
        # 3.1. Allocate memory for separation matrix, spike/noise thresholds and ICs,
        # and initialize spikes and SIL scores
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
        spikes_t_tmp = []
        sil_scores_tmp = []

        # 3.2. Run ICA loop
        w_init = self._initialize_weights(emg_white)
        for i in range(self._n_mu, self._n_mu_target):
            logging.info(f"----- IC {i + 1} -----")

            # Kernel-Constrained FastICA step
            w_i = self._ker_fast_ica_iter(emg_white, mu_subspace=sep_mtx[:i])
            ic_i = w_i @ emg_white
            if self._square_ics:  # square
                ic_i **= 2
            else:  # solve sign uncertainty using skewness
                if (ic_i**3).mean() < 0:
                    w_i *= -1

            # Threshold selection
            spike_th_i, spikes_i = self._thresh_cov(ic_i, pp=200)

            # Correlation-Constrained FastICA step
            w_i = self._corr_fast_ica_iter(emg_white, spikes_i)

            # Re-select threshold
            spike_th_i, spikes_i = self._thresh_cov(ic_i, pp=200)

            # Save separation vector and spike/noise threshold
            sep_mtx[i] = w_i
            spike_ths[i] = spike_th_i
            # Save current IC, discharge times and SIL
            ics[i] = (w_i @ emg_white) ** 2 if self._square_ics else w_i @ emg_white
            spikes_t_tmp.append(spikes_i / self._fs)
            sil_scores_tmp.append(sil)

        # 4. Post-processing
        # 4.1. SIL, CoV-ISI, CoV-Amp, and DR thresholding
        idx_to_keep = list(range(self._n_mu))
        for i in range(self._n_mu, self._n_mu_target):
            # Check SIL
            sil = sil_scores_tmp[i]
            if np.isnan(sil) or sil <= self._sil_th:
                logging.info(
                    f"{i}-th IC: SIL below threshold (SIL = {sil:.3f} <= {self._sil_th:.3f}) -> skipped."
                )
                continue

            # Check CoV-ISI
            cov_isi = spike_stats.cov_isi(spikes_t_tmp[i], self._cov_isi_rest)
            if np.isnan(cov_isi) or cov_isi >= self._cov_isi_th:
                logging.info(
                    f"{i}-th IC: CoV-ISI above threshold (CoV-ISI = {cov_isi:.2%} >= {self._cov_isi_th:.2%})"
                    f" -> skipped."
                )
                continue

            # Check CoV-Amp
            cov_amp = spike_stats.cov_amp(
                ics[i, (spikes_t_tmp[i] * self._fs).astype(np.int64)].cpu().numpy()
            )
            if np.isnan(cov_amp) or cov_amp >= self._cov_amp_th:
                logging.info(
                    f"{i}-th IC: CoV-Amp above threshold (CoV-Amp = {cov_amp:.2%} >= {self._cov_amp_th:.2%})"
                    f" -> skipped."
                )
                continue

            # Check discharge rate
            avg_dr = spike_stats.instantaneous_discharge_rate(spikes_t_tmp[i]).mean()
            if avg_dr <= self._min_dr:
                logging.info(
                    f"{i}-th IC: discharge rate below threshold (DR = {avg_dr:.3f} <= {self._min_dr:.3f})"
                    f" -> skipped."
                )
                continue
            if avg_dr >= self._max_dr:
                logging.info(
                    f"{i}-th IC: discharge rate above threshold (DR = {avg_dr:.3f} >= {self._max_dr:.3f})"
                    f" -> skipped."
                )
                continue

            logging.info(
                f"{i}-th IC: SIL = {sil:.3f}, CoV-ISI = {cov_isi:.2%}, "
                f"CoV-Amp = {cov_amp:.2%}, DR = {avg_dr:.3f} -> accepted."
            )
            idx_to_keep.append(i)
        # Keep only valid entries
        sep_mtx = sep_mtx[idx_to_keep]
        spike_ths = spike_ths[idx_to_keep]
        ics = ics[idx_to_keep]
        # Turn lists into dictionaries
        spikes_t = {}
        sil_scores = {}
        for i, idx in enumerate(idx_to_keep):
            spikes_t[f"MU{i}"] = spikes_t_tmp[idx]
            sil_scores[i] = sil_scores_tmp[idx]

        logging.info(f"Extracted {len(spikes_t)} MUs before replicas removal.")

        # 4.2. Replicas removal
        logging.info("Looking for delayed replicas...")
        ics_bin = utils.sparse_to_dense(spikes_t, n_samp / self._fs, self._fs)
        duplicate_mus = utils.find_replicas(
            ics_bin, fs=self._fs, tol_ms=1, min_perc=0.3
        )
        idx_to_keep = list(range(len(spikes_t)))
        for main_mu, dup_mus in duplicate_mus.items():
            # Unify duplicate MUs
            dup_mus = [main_mu] + dup_mus
            dup_str = ", ".join([f"{mu}" for mu in dup_mus])
            logging.info(f"Found group of duplicate MUs: {dup_str}.")

            # Keep only the MU with the highest SIL
            sil_dup = {k: v for k, v in sil_scores.items() if k in dup_mus}
            mu_keep = max(sil_dup, key=lambda k: sil_dup[k])
            logging.info(f"Keeping MU {mu_keep} (SIL = {sil_dup[mu_keep]:.2%}).")

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
        """
        Decompose the given EMG signal into MUAPTs using the frozen decomposition model.

        Parameters
        ----------
        emg : Signal
            EMG signal with shape (n_samples, n_channels).

        Returns
        -------
        DataFrame
            A DataFrame with shape (n_samples, n_mu) containing the components estimated by ICA.
        dict of str: ndarray
            Dictionary containing the discharge times for each MU.

        Raises
        ------
        AttributeError
            If the decomposition model is not trained.
        """
        if self._sep_mtx is None or self._spike_ths is None:
            raise AttributeError(
                'The decomposition model is not trained, call "decompose_training(x)" first.'
            )

        # 1. Extension
        emg_ext = preprocessing.extend_signal(emg, self._f_ext)
        n_samp = emg_ext.shape[0]
        emg_ext = torch.from_numpy(emg_ext).to(self._device)

        # 2. Whitening + ICA
        ics = self._whiten_model.whiten_inference(emg_ext) @ self._sep_mtx.T
        if self._square_ics:
            ics **= 2

        # 3. Spike extraction
        spikes_t = {}
        for i in range(self._n_mu):
            spikes_i = utils.detect_spikes(
                ics[:, i],
                ref_period=self._ref_period,
                threshold=self._spike_ths[i].item(),
                prng=self._prng,
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

    def _ker_fast_ica_iter(
        self, x_w: torch.Tensor, mu_subspace: torch.Tensor, w_i_init: torch.Tensor
    ) -> torch.Tensor:
        """Kernel-Constrained FastICA iteration."""
        # Project data onto the null space
        if mu_subspace.size(0) == 0:
            k = None
            x_r = x_w
        else:
            k = _null_space(mu_subspace)
            x_r = k.T @ x_w

        # Initialize weight
        w_i = w_i_init
        w_i /= torch.linalg.norm(w_i)

        # FastICA iterations
        iter_idx = 1
        while iter_idx <= self._max_iter:
            g_res = self._g_func(w_i @ x_r)
            w_i_new = (x_r * g_res.g1_u).mean(dim=1) - g_res.g2_u.mean() * w_i
            w_i_new /= torch.linalg.norm(w_i_new)

            distance = 1 - abs((w_i_new @ w_i).item())
            w_i = w_i_new
            if distance < self._conv_th:
                logging.info(
                    f"FastICA converged after {iter_idx} iterations, the distance is: {distance:.3e}."
                )
                break
            iter_idx += 1

        # Project back to the original space
        if k is not None:
            w_i = k @ w_i
            w_i /= torch.linalg.norm(w_i)

        return w_i

    def _accept_mu(self) -> bool:
        """Decide whether to accept the current MU based on validation metrics."""
        # Check SIL
        sil = sil_scores_tmp[i]
        if np.isnan(sil) or sil <= self._sil_th:
            logging.info(
                f"{i}-th IC: SIL below threshold (SIL = {sil:.3f} <= {self._sil_th:.3f}) -> skipped."
            )
            return False

        # Check CoV-ISI
        cov_isi = spike_stats.cov_isi(spikes_t_tmp[i], self._cov_isi_rest)
        if np.isnan(cov_isi) or cov_isi >= self._cov_isi_th:
            logging.info(
                f"{i}-th IC: CoV-ISI above threshold (CoV-ISI = {cov_isi:.2%} >= {self._cov_isi_th:.2%})"
                f" -> skipped."
            )
            return False

        # Check CoV-Amp
        cov_amp = spike_stats.cov_amp(
            ics[i, (spikes_t_tmp[i] * self._fs).astype(np.int64)].cpu().numpy()
        )
        if np.isnan(cov_amp) or cov_amp >= self._cov_amp_th:
            logging.info(
                f"{i}-th IC: CoV-Amp above threshold (CoV-Amp = {cov_amp:.2%} >= {self._cov_amp_th:.2%})"
                f" -> skipped."
            )
            return False

        # Check discharge rate
        avg_dr = spike_stats.instantaneous_discharge_rate(spikes_t_tmp[i]).mean()
        if avg_dr <= self._min_dr or avg_dr >= self._max_dr:
            logging.info(
                f"{i}-th IC: discharge rate outside range (DR = {avg_dr:.3f} !in [{self._min_dr:.3f}, {self._max_dr:.3f}])"
                f" -> skipped."
            )
            return False

        logging.info(
            f"{i}-th IC: SIL = {sil:.3f}, CoV-ISI = {cov_isi:.2%}, "
            f"CoV-Amp = {cov_amp:.2%}, DR = {avg_dr:.3f} -> accepted."
        )
        return True

    def _corr_fast_ica_iter(
        self, x_w: torch.Tensor, spikes_i: torch.Tensor
    ) -> torch.Tensor:
        """Constrained FastICA iteration."""
        n_samp = x_w.size(1)

        # Initialize w_i
        r = x_w @ spikes_i
        r /= torch.linalg.norm(r)
        w_i = r.clone()
        mu = 0.3 * n_samp

        # FastICA iterations
        iter_idx = 1
        while iter_idx <= self._max_iter:
            g_res = self._g_func(w_i @ x_w)
            w_i_new = g_res.g2_u.mean() * w_i - (x_w * g_res.g1_u).mean(dim=1) + mu * r
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

    def _thresh_cov(self, ic_i: torch.Tensor, pp=200) -> tuple[float, torch.Tensor]:
        """
        Automatic spike threshold selection via ISI CoV minimization.
        """
        ic_i_ = ic_i.cpu().numpy()
        b = np.sort(ic_i_)[::-1]

        # threshold candidates
        t = np.linspace(2.0, b[19], pp)

        cov = np.ones_like(t)
        for i, th in enumerate(t):
            peaks, _ = signal.find_peaks(ic_i_, height=th, distance=self._ref_period)
            if len(peaks) < 2:
                cov[i] = np.inf
                continue

            isi = np.diff(peaks)
            cov[i] = np.std(isi) / np.mean(isi)

        # ignore tiny CoV values
        valid = cov > 0.05
        idx = np.argmin(cov[valid])
        spike_th = t[valid][idx]

        # final spike train
        spikes = torch.zeros_like(ic_i, dtype=torch.uint8)
        peaks, _ = signal.find_peaks(ic_i_, height=spike_th, distance=self._ref_period)
        spikes[peaks] = 1

        return spike_th, spikes
