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
from math import sqrt

import numpy as np
import torch

from .._base import Signal, signal_to_tensor
from ..utils import eigendecomposition
from ._abc_whitening import WhiteningModel


def pca_whitening(
    x: Signal,
    n_pcs: int | str = "auto",
    keep_dim: bool = False,
    solver: str = "svd",
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, PCAWhitening]:
    """Function performing PCA whitening.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    n_pcs : int or str, default="auto"
        Number of components to be selected:
        - if set to the string "auto", it will be chosen automatically based on the average of the smallest
        half of eigenvalues/singular values;
        - if set to the string "all", all components will be retained;
        - otherwise, it will be set to the given number.
    keep_dim : bool, default=False
        Whether to re-project the low-dimensional whitened data to the original dimensionality.
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or str or None, default=None
        Torch device.

    Returns
    -------
    Tensor
        Whitened signal with shape (n_samples, n_components).
    PCAWhitening
        Fit PCA whitening model.

    Raises
    ------
    TypeError
        If the input is neither an array, a DataFrame nor a Tensor.
    ValueError
        If the input is not 2D.
    """
    whiten_model = PCAWhitening(n_pcs, keep_dim, solver, device)
    x_w = whiten_model.fit_transform(x)

    return x_w, whiten_model


class PCAWhitening(WhiteningModel):
    """Class implementing PCA whitening.

    Parameters
    ----------
    n_pcs : int or str, default="auto"
        Number of components to be selected:
        - if set to the string "auto", it will be chosen automatically based on the average of the smallest
        half of eigenvalues/singular values;
        - if set to the string "all", all components will be retained;
        - otherwise, it will be set to the given number.
    keep_dim : bool, default=False
        Whether to re-project the low-dimensional whitened data to the original dimensionality.
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or str or None, default=None
        Torch device.

    Attributes
    ----------
    _keep_dim : bool
        Whether to re-project the low-dimensional whitened data to the original dimensionality.
    _solver : str
        The solver used for whitening, either "svd" (default) or "eigh".
    _device : device or None
        Torch device.
    """

    def __init__(
        self,
        n_pcs: int | str = "auto",
        keep_dim: bool = False,
        solver: str = "svd",
        device: torch.device | str | None = None,
    ) -> None:
        assert (isinstance(n_pcs, int) and n_pcs > 0) or (
            isinstance(n_pcs, str) and n_pcs in ("auto", "all")
        ), 'n_pcs must be either a positive integer, "auto" or "all".'
        assert solver in ("svd", "eigh"), 'The solver must be either "svd" or "eigh".'

        logging.info(f'Instantiating PCAWhitening using "{solver}" solver.')

        # Map "auto" -> -1 and "all" -> 0
        if n_pcs == "auto":
            self._n_pcs = -1
        elif n_pcs == "all":
            self._n_pcs = 0
        else:
            self._n_pcs = n_pcs

        self._keep_dim = keep_dim
        self._solver = solver
        self._device = torch.device(device) if isinstance(device, str) else device

    @property
    def eig_vecs(self) -> torch.Tensor:
        """Tensor: Property for getting the matrix of eigenvectors."""
        return self._eig_vecs

    @property
    def eig_vals(self) -> torch.Tensor:
        """Tensor: Property for getting the vector of eigenvalues."""
        return self._eig_vals

    @property
    def mean_vec(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated mean vector."""
        return self._mean_vec

    @property
    def white_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated whitening matrix."""
        return self._white_mtx

    @property
    def n_pcs(self) -> int:
        """int: Property for getting the number of principal components."""
        return self._n_pcs

    @property
    def exp_var_ratio(self) -> np.ndarray:
        """Tensor: Property for getting the vector of explained variance ratio."""
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
        is_fit = hasattr(self, "_mean_vec") and hasattr(self, "_white_mtx")
        assert is_fit, "Fit the model first."

        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device).T

        # Center and whiten signal
        x_tensor -= self._mean_vec
        x_w = self._white_mtx @ x_tensor

        return x_w.T

    def _fit_transform(self, x: Signal) -> torch.Tensor:
        """Helper method for fit and fit_transform."""
        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device).T
        n_ch, n_samp = x_tensor.size()
        self._mean_vec = x_tensor.mean(dim=1, keepdim=True)
        x_tensor -= self._mean_vec

        # Compute eigenvectors and eigenvalues of the covariance matrix X @ X.T / n_samp
        if self._solver == "svd":
            # The left-singular vectors of X are the eigenvectors of X @ X.T
            # The singular values of X are the square root of the eigenvalues of X @ X.T
            self._eig_vecs, s_vals, _ = torch.linalg.svd(x_tensor, full_matrices=False)
            self._eig_vals = s_vals**2

            exp_var_ratio = (self._eig_vals / self._eig_vals.sum()).cpu().numpy()
            d_mtx = torch.diag(1.0 / s_vals) * sqrt(n_samp)
        else:
            n_samp = x_tensor.size(1)
            cov_mtx = x_tensor @ x_tensor.T / n_samp
            self._eig_vecs, self._eig_vals = eigendecomposition(cov_mtx)

            exp_var_ratio = (self._eig_vals / self._eig_vals.sum()).cpu().numpy()
            d_mtx = torch.diag(1.0 / torch.sqrt(self._eig_vals))
        self._eig_vecs *= torch.sign(self._eig_vecs[0])  # guarantee consistent sign

        # Select number of components to retain
        if self._n_pcs < 0:  # automatic selection
            rank_th = self._eig_vals[self._eig_vals.size(0) // 2 :].mean()
            self._n_pcs = int(torch.sum(torch.ge(self._eig_vals, rank_th)).item())
        elif self._n_pcs == 0:
            self._n_pcs = n_ch
        assert (
            n_ch >= self._n_pcs
        ), f"Too few channels ({n_ch}) with respect to target components ({self._n_pcs})."

        logging.info(f"Reducing dimension of data from {n_ch} to {self._n_pcs}.")
        d_mtx = d_mtx[: self._n_pcs, : self._n_pcs]
        self._eig_vecs = self._eig_vecs[:, : self._n_pcs]
        self._exp_var_ratio = exp_var_ratio[: self._n_pcs]

        self._white_mtx = d_mtx @ self._eig_vecs.T
        if self._keep_dim:  # re-project to original dimensionality
            self._white_mtx = self._eig_vecs @ self._white_mtx
            logging.info(f"Re-projecting dimensionality to {self._white_mtx.size(0)}.")
        x_w = self._white_mtx @ x_tensor

        return x_w.T
