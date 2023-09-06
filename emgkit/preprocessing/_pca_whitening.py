"""Function and class implementing the PCA whitening algorithm.


Copyright 2022 Mattia Orlandi

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

import numpy as np
import torch

from .._base import Signal, signal_to_tensor
from ._abc_whitening import WhiteningModel


def pca_whitening(
    x: Signal,
    n_pcs: int | str = -1,
    solver: str = "svd",
    device: torch.device | None = None,
) -> tuple[np.ndarray | torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function performing PCA whitening.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    n_pcs : int or str, default=-1
        Number of components to be selected:
        - if set to the string "auto", it will be chosen automatically based on the average of the smallest
        half of eigenvalues/singular values;
        - if zero or negative, all components will be retained.
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or None, default=None
        Torch device.

    Returns
    -------
    Tensor
        Whitened signal with shape (n_samples, n_components).
    Tensor
        Estimated mean vector.
    Tensor
        Estimated whitening matrix.

    Raises
    ------
    TypeError
        If the input is neither an array, a DataFrame nor a Tensor.
    ValueError
        If the input is not 2D.
    """
    whiten_model = PCAWhitening(n_pcs, solver, device)
    x_w = whiten_model.fit_transform(x)

    return x_w, whiten_model.mean_vec, whiten_model.white_mtx


class PCAWhitening(WhiteningModel):
    """Class implementing PCA whitening.

    Parameters
    ----------
    n_pcs : int or str, default=-1
        Number of components to be selected:
        - if set to the string "auto", it will be chosen automatically based on the average of the smallest
        half of eigenvalues/singular values;
        - if zero or negative, all components will be retained.
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or None, default=None
        Torch device.

    Attributes
    ----------
    _solver : str
        The solver used for whitening, either "svd" (default) or "eigh".
    _device : device or None
        Torch device.
    """

    def __init__(
        self,
        n_pcs: int | str = -1,
        solver: str = "svd",
        device: torch.device | None = None,
    ) -> None:
        assert isinstance(n_pcs, int) or (
            isinstance(n_pcs, str) and n_pcs == "auto"
        ), 'n_pcs must be either an integer or "auto".'
        assert solver in ("svd", "eigh"), 'The solver must be either "svd" or "eigh".'

        logging.info(f'Instantiating PCAWhitening using "{solver}" solver.')

        self._n_pcs: int | str = n_pcs
        self._solver: str = solver
        self._device: torch.device | None = device

        self._exp_var_ratio: np.ndarray | None = None
        self._mean_vec: torch.Tensor | None = None
        self._white_mtx: torch.Tensor | None = None

    @property
    def mean_vec(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated mean vector."""
        return self._mean_vec

    @property
    def white_mtx(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated whitening matrix."""
        return self._white_mtx

    @property
    def n_pcs(self) -> int:
        """int: Property for getting the number of principal components."""
        return self._n_pcs

    @property
    def exp_var_ratio(self) -> np.ndarray | None:
        """Tensor or None: Property for getting the vector of explained variance ratio."""
        return self._exp_var_ratio

    def fit(self, x: Signal) -> WhiteningModel:
        """Fit the whitening model on the given signal.

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).

        Returns
        -------
        WhiteningModel
            The fitted whitening model.

        Raises
        ------
        TypeError
            If the input is neither an array, a DataFrame nor a Tensor.
        ValueError
            If the input is not 2D.
        """
        self._fit_transform(x)
        return self

    def fit_transform(self, x: Signal) -> torch.Tensor:
        """Fit the whitening model on the given signal and return the whitened signal.

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).

        Returns
        -------
        Tensor
            Whitened signal with shape (n_samples, n_components).

        Raises
        ------
        TypeError
            If the input is neither an array, a DataFrame nor a Tensor.
        ValueError
            If the input is not 2D.
        """
        return self._fit_transform(x)

    def transform(self, x: Signal) -> torch.Tensor:
        """Whiten the given signal using the fitted whitening model.

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).

        Returns
        -------
        Tensor
            Whitened signal with shape (n_samples, n_components).

        Raises
        ------
        TypeError
            If the input is neither an array, a DataFrame nor a Tensor.
        ValueError
            If the input is not 2D.
        """
        assert (
            self._mean_vec is not None and self._white_mtx is not None
        ), "Mean vector or whitening matrix are null, fit the model first."

        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device, allow_1d=False).T

        # Center and whiten signal
        x_tensor -= self._mean_vec
        x_w = self._white_mtx @ x_tensor

        return x_w.T

    def _fit_transform(self, x: Signal) -> torch.Tensor:
        """Helper method for fit and fit_transform."""
        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device, allow_1d=False).T
        n_ch, n_samp = x_tensor.size()
        self._mean_vec = x_tensor.mean(dim=1, keepdim=True)
        x_tensor -= self._mean_vec

        if self._solver == "svd":
            e, d, _ = torch.linalg.svd(x_tensor, full_matrices=False)

            d_sq = d**2  # singular values are the square root of eigenvalues
            exp_var_ratio = (d_sq / d_sq.sum()).cpu().numpy()
            d_mtx = torch.diag(1.0 / d) * sqrt(n_samp - 1)
        else:
            d, e = torch.linalg.eigh(torch.cov(x_tensor))

            # Improve numerical stability
            eps = torch.finfo(d.dtype).eps
            degenerate_idx = torch.lt(d, eps).nonzero()
            if torch.any(degenerate_idx):
                warnings.warn(
                    f'Some eigenvalues are smaller than epsilon ({eps:.3e}), try using "SVD" solver.'
                )
            d[degenerate_idx] = eps

            sort_idx = torch.argsort(d, descending=True)
            d, e = d[sort_idx], e[:, sort_idx]
            exp_var_ratio = (d / d.sum()).cpu().numpy()
            d_mtx = torch.diag(1.0 / torch.sqrt(d))
        e *= torch.sign(e[0])  # guarantee consistent sign

        # Select number of components to retain
        if self._n_pcs == "auto":
            rank_th = d[d.size(0) // 2 :].mean()
            self._n_pcs = torch.sum(torch.ge(d, rank_th))
        elif self._n_pcs <= 0:
            self._n_pcs = n_ch
        logging.info(f"Reducing dimension of data from {n_ch} to {self._n_pcs}.")
        d_mtx = d_mtx[: self._n_pcs, : self._n_pcs]
        e = e[:, : self._n_pcs]
        self._exp_var_ratio = exp_var_ratio[: self._n_pcs]

        self._white_mtx = d_mtx @ e.T
        x_w = self._white_mtx @ x_tensor

        return x_w.T
