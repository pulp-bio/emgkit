"""Function and class implementing the FastICA algorithm
(https://doi.org/10.1016/S0893-6080(00)00026-5).


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
import warnings
from math import sqrt

import torch

from .._base import Signal, signal_to_tensor
from ..preprocessing import pca_whitening, zca_whitening
from . import contrast_functions as cf
from ._abc_ica import ICA
from ._warn import ConvergenceWarning


def fast_ica(
    x: Signal,
    n_ics: int | str = "all",
    whiten_alg: str = "zca",
    strategy: str = "symmetric",
    g_name: str = "logcosh",
    conv_th: float = 1e-4,
    max_iter: int = 200,
    do_saddle_test: bool = False,
    device: torch.device | str | None = None,
    seed: int | None = None,
    w_init: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function implementing FastICA.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    n_ics : int or str, default="all"
        Number of components to estimate:
        - if set to the string "all", it will be set to the number of channels in the signal;
        - otherwise, it will be set to the given number.
    whiten_alg : {"zca", "pca", "none"}, default="zca"
        Whitening algorithm.
    strategy : {"symmetric", "deflation"}, default="symmetric"
        Name of FastICA strategy.
    g_name : {"logcosh", "gauss", "kurtosis", "skewness", "rati"}, default="logcosh"
        Name of the contrast function.
    conv_th : float, default=1e-4
        Threshold for convergence.
    max_iter : int, default=200
        Maximum n. of iterations.
    do_saddle_test : bool, default=False
        Whether to perform the test of saddle points.
    device : device or str or None, default=None
        Torch device.
    seed : int or None, default=None
        Seed for the internal PRNG.
    w_init : Tensor or None, default=None
        Initial separation matrix with shape (n_components, n_channels).
    **kwargs
        Keyword arguments forwarded to whitening algorithm.

    Returns
    -------
    Tensor
        Estimated source signal with shape (n_samples, n_components).
    Tensor
        Estimated mean vector.
    Tensor
        Estimated separation matrix with shape (n_components, n_channels).

    Warns
    -----
    ConvergenceWarning
        The algorithm didn't converge.
    """
    ica_model = FastICA(
        n_ics,
        whiten_alg,
        strategy,
        g_name,
        conv_th,
        max_iter,
        do_saddle_test,
        device,
        seed,
        **kwargs,
    )
    ics = ica_model.fit_transform(x, w_init)

    return ics, ica_model.mean_vec, ica_model.sep_mtx


class FastICA(ICA):
    """Class implementing FastICA.

    Parameters
    ----------
    n_ics : int or str, default="all"
        Number of components to estimate:
        - if set to the string "all", it will be set to the number of channels in the signal;
        - otherwise, it will be set to the given number.
    whiten_alg : {"zca", "pca", "none"}, default="zca"
        Whitening algorithm.
    strategy : {"symmetric", "deflation"}, default="symmetric"
        Name of FastICA strategy.
    g_name : {"logcosh", "gauss", "kurtosis", "skewness", "rati"}, default="logcosh"
        Name of the contrast function.
    conv_th : float, default=1e-4
        Threshold for convergence.
    max_iter : int, default=200
        Maximum n. of iterations.
    do_saddle_test : bool, default=False
        Whether to perform the test of saddle points (relevant for symmetric strategy).
    device : device or str or None, default=None
        Torch device.
    seed : int or None, default=None
        Seed for the internal PRNG.
    **kwargs
        Keyword arguments forwarded to whitening algorithm.

    Attributes
    ----------
    _n_ics : int
        Number of components to estimate.
    _whiten_alg : str
        Whitening algorithm.
    _whiten_kw : dict
        Whitening arguments.
    _strategy : str
        Name of FastICA strategy.
    _g_func : ContrastFunction
        Contrast function.
    _conv_th : float
        Threshold for convergence.
    _max_iter : int
        Maximum n. of iterations.
    _do_saddle_test : bool
        Whether to perform the test of saddle points (relevant for symmetric strategy).
    _device : device or None
        Torch device.
    """

    def __init__(
        self,
        n_ics: int | str = "all",
        whiten_alg: str = "zca",
        strategy: str = "symmetric",
        g_name: str = "logcosh",
        conv_th: float = 1e-4,
        max_iter: int = 200,
        do_saddle_test: bool = False,
        device: torch.device | str | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        assert (isinstance(n_ics, int) and n_ics > 0) or (
            isinstance(n_ics, str) and n_ics == "all"
        ), 'n_ics must be either a positive integer or "all".'
        assert whiten_alg in (
            "zca",
            "pca",
            "none",
        ), f'Whitening can be either "zca", "pca" or "none": the provided one was "{whiten_alg}".'
        assert strategy in (
            "symmetric",
            "deflation",
        ), f'Strategy can be either "symmetric" or "deflation": the provided one was "{strategy}".'
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

        logging.info(
            f'Instantiating FastICA using "{strategy}" strategy and "{g_name}" contrast function.'
        )

        # Map "all" -> 0
        self._n_ics = 0 if n_ics == "all" else n_ics
        self._whiten_alg = whiten_alg
        self._whiten_kw = kwargs
        self._strategy = strategy
        g_dict = {
            "logcosh": cf.logcosh,
            "gauss": cf.gauss,
            "kurtosis": cf.kurtosis,
            "skewness": cf.skewness,
            "rati": cf.rati,
        }
        self._g_func = g_dict[g_name]
        self._conv_th = conv_th
        self._max_iter = max_iter
        self._do_saddle_test = do_saddle_test
        self._device = torch.device(device) if isinstance(device, str) else device

        if seed is not None:
            torch.manual_seed(seed)

    @property
    def mean_vec(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated mean vector."""
        return self._mean_vec

    @property
    def sep_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated separation matrix."""
        return self._sep_mtx

    def fit(
        self,
        x: Signal,
        w_init: torch.Tensor | None = None,
    ) -> ICA:
        """Fit the ICA model on the given signal.

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).
        w_init : Tensor or None, default=None
            Initial separation matrix with shape (n_components, n_channels).

        Returns
        -------
        ICA
            The fitted ICA model.

        Warns
        -----
        ConvergenceWarning
            The algorithm didn't converge.
        """
        self._fit_transform(x, w_init)
        return self

    def fit_transform(
        self,
        x: Signal,
        w_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fit the ICA model on the given signal and return the estimated ICs.

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).
        w_init : Tensor or None, default=None
            Initial separation matrix with shape (n_components, n_channels).

        Returns
        -------
        Tensor
            Estimated source signal with shape (n_samples, n_components).

        Warns
        -----
        ConvergenceWarning
            The algorithm didn't converge.
        """
        return self._fit_transform(x, w_init)

    def transform(self, x: Signal) -> torch.Tensor:
        """Decompose the given signal using the fitted ICA model.

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).

        Returns
        -------
        Tensor
            Estimated source signal with shape (n_samples, n_components).
        """
        assert (
            self._mean_vec is not None and self._sep_mtx is not None
        ), "Mean vector or separation matrix are null, fit the model first."

        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device, allow_1d=False).T

        # Decompose signal
        ics = self._sep_mtx @ (x_tensor - self._mean_vec)

        return ics.T

    def _fit_transform(self, x: Signal, w_init: torch.Tensor | None) -> torch.Tensor:
        """Helper method for fit and fit_transform."""
        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device, allow_1d=False)

        # Whitening
        if self._whiten_alg != "none":
            whiten_alg_dict = {"zca": zca_whitening, "pca": pca_whitening}
            x_tensor, self._mean_vec, white_mtx = whiten_alg_dict[self._whiten_alg](
                x_tensor, **self._whiten_kw, device=self._device
            )
        else:
            x_tensor = x_tensor
            self._mean_vec = torch.zeros(
                x_tensor.size(0), 1, dtype=x_tensor.dtype, device=self._device
            )
            white_mtx = torch.eye(
                x_tensor.size(1), dtype=x_tensor.dtype, device=self._device
            )
        x_tensor = x_tensor.T

        n_ch = x_tensor.size(0)
        if self._n_ics == 0:
            self._n_ics = n_ch
        assert (
            n_ch >= self._n_ics
        ), f"Too few channels ({n_ch}) with respect to target components ({self._n_ics})."

        if w_init is None:
            w_init = torch.randn(
                self._n_ics, n_ch, dtype=x_tensor.dtype, device=self._device
            )
        else:
            assert w_init.shape == (
                self._n_ics,
                n_ch,
            ), f"The shape of w_init should be ({self._n_ics}, {n_ch})."
            w_init = w_init.to(self._device)

        # Perform decomposition
        strategy_dict = {"symmetric": self._symmetric, "deflation": self._deflation}
        self._sep_mtx = strategy_dict[self._strategy](x_tensor, w_init)
        ics = self._sep_mtx @ x_tensor
        self._sep_mtx = self._sep_mtx @ white_mtx

        return ics.T

    def _symmetric(self, x: torch.Tensor, w_init: torch.Tensor) -> torch.Tensor:
        """Helper method for symmetric algorithm."""
        n_samp = x.size(1)

        def sym_orth(w_: torch.Tensor) -> torch.Tensor:
            eig_vals, eig_vecs = torch.linalg.eigh(w_ @ w_.T)

            # Improve numerical stability
            eig_vals = torch.clip(eig_vals, min=torch.finfo(w_.dtype).tiny)

            d_mtx = torch.diag(1.0 / torch.sqrt(eig_vals))
            return eig_vecs @ d_mtx @ eig_vecs.T @ w_

        w = sym_orth(w_init)

        saddle_test_done = False
        max_iter = self._max_iter
        rot_mtx = 1 / torch.as_tensor(
            [[sqrt(2), -sqrt(2)], [sqrt(2), sqrt(2)]],
            dtype=x.dtype,
            device=self._device,
        )
        rotated = torch.zeros(self._n_ics, dtype=torch.bool)
        while True:
            iter_idx = 1
            converged = False
            while iter_idx <= max_iter:
                g_res = self._g_func(w @ x)
                w_new = (
                    g_res.g1_u @ x.T / n_samp - g_res.g2_u.mean(dim=1, keepdim=True) * w
                )
                w_new = sym_orth(w_new)

                # Compute distance:
                # 1. Compute absolute dot product between old and new separation vectors (i.e., the rows of W)
                distance = torch.abs(torch.einsum("ij,ij->i", w, w_new))
                # 2. Absolute dot product should be close to 1, thus subtract 1 and take absolute value
                distance = torch.abs(distance - 1)
                # 3. Consider maximum distance
                distance = torch.max(distance).item()
                logging.info(f"FastICA iteration {iter_idx}: {distance:.3e}.")

                w = w_new

                if distance < self._conv_th:
                    converged = True
                    logging.info(
                        f"FastICA converged after {iter_idx} iterations, the distance is: {distance:.3e}."
                    )
                    break

                iter_idx += 1

            if saddle_test_done:
                break
            if not converged:
                warnings.warn("FastICA didn't converge.", ConvergenceWarning)
            if not self._do_saddle_test:
                break

            logging.info("Performing saddle test...")
            ics = w @ x
            ics_g_ret = self._g_func(ics)
            ics_score = (ics_g_ret.g_u.mean(dim=1) - ics_g_ret.g_nu) ** 2
            # Check each pair that has not already been rotated
            positive = False
            for i in range(self._n_ics):
                for j in range(i + 1, self._n_ics):
                    if torch.all(~rotated[[i, j]]):
                        # Rotate pair and compute score
                        rot_ics = rot_mtx @ ics[[i, j]]
                        rot_ics_g_ret = self._g_func(rot_ics)
                        rot_ics_score = (
                            rot_ics_g_ret.g_u.mean(dim=1) - rot_ics_g_ret.g_nu
                        ) ** 2
                        # If the score of rotated ICs is higher, apply rotation
                        if rot_ics_score.max() > ics_score[[i, j]].max():
                            w[[i, j]] = rot_mtx @ w[[i, j]]
                            rotated[[i, j]] = True
                            positive = True

            if positive:
                logging.info(
                    "Some ICs were found to be positive at saddle point test, refining..."
                )
                saddle_test_done = True
                max_iter = 2
            else:
                logging.info("Saddle point test ok.")
                break

        return w

    def _deflation(self, x: torch.Tensor, w_init: torch.Tensor) -> torch.Tensor:
        """Helper method for deflation algorithm."""
        w = w_init.clone()

        failed_convergence = False
        for i in range(self._n_ics):
            logging.info(f"----- IC {i + 1} -----")

            w_i = w[i]
            w_i /= torch.linalg.norm(w_i)

            iter_idx = 1
            converged = False
            while iter_idx <= self._max_iter:
                g_res = self._g_func(w_i @ x)
                w_i_new = (x * g_res.g1_u).mean(dim=1) - g_res.g2_u.mean() * w_i
                w_i_new -= w_i_new @ w[:i].T @ w[:i]  # Gram-Schmidt decorrelation
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

            if not converged:
                logging.info("FastICA didn't converge for current component.")
                failed_convergence = True

            w[i] = w_i

        if failed_convergence:
            warnings.warn(
                "FastICA didn't converge for at least one component.",
                ConvergenceWarning,
            )

        return w
