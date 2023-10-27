"""Function and class implementing the ZCA whitening algorithm.


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

import torch

from .._base import Signal, signal_to_tensor
from ..utils import eigendecomposition
from ._abc_whitening import WhiteningModel


def zca_whitening(
    x: Signal,
    solver: str = "svd",
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, ZCAWhitening]:
    """Function performing ZCA whitening.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or str or None, default=None
        Torch device.

    Returns
    -------
    Tensor
        Whitened signal with shape (n_samples, n_components).
    ZCAWhitening
        Fit ZCA whitening model.

    Raises
    ------
    TypeError
        If the input is neither an array, a DataFrame nor a Tensor.
    ValueError
        If the input is not 2D.
    """
    whiten_model = ZCAWhitening(solver, device)
    x_w = whiten_model.fit_transform(x)

    return x_w, whiten_model


class ZCAWhitening(WhiteningModel):
    """Class implementing ZCA whitening.

    Parameters
    ----------
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or str or None, default=None
        Torch device.

    Attributes
    ----------
    _solver : str
        The solver used for whitening, either "svd" (default) or "eigh".
    _device : device or None
        Torch device.
    """

    def __init__(
        self, solver: str = "svd", device: torch.device | str | None = None
    ) -> None:
        assert solver in ("svd", "eigh"), 'The solver must be either "svd" or "eigh".'

        logging.info(f'Instantiating ZCAWhitening using "{solver}" solver.')

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
        # Fit the model and return self
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
        # Fit the model and return result
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
        x_tensor = signal_to_tensor(x, self._device).T

        # Center and whiten signal
        x_tensor -= self._mean_vec
        x_w = self._white_mtx @ x_tensor

        return x_w.T

    def _fit_transform(self, x: Signal) -> torch.Tensor:
        """Helper method for fit and fit_transform."""
        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device).T
        n_samp = x_tensor.size(1)
        self._mean_vec = x_tensor.mean(dim=1, keepdim=True)
        x_tensor -= self._mean_vec

        # Compute eigenvectors and eigenvalues of the covariance matrix X @ X.T / n_samp
        if self._solver == "svd":
            # The left-singular vectors of X are the eigenvectors of X @ X.T
            # The singular values of X are the square root of the eigenvalues of X @ X.T
            self._eig_vecs, s_vals, _ = torch.linalg.svd(x_tensor, full_matrices=False)
            self._eig_vals = s_vals**2

            d_mtx = torch.diag(1.0 / s_vals) * sqrt(n_samp)
        else:
            n_samp = x_tensor.size(1)
            cov_mtx = x_tensor @ x_tensor.T / n_samp
            self._eig_vecs, self._eig_vals = eigendecomposition(cov_mtx)

            d_mtx = torch.diag(1.0 / torch.sqrt(self._eig_vals))
        self._eig_vecs *= torch.sign(self._eig_vecs[0])  # guarantee consistent sign

        self._white_mtx = self._eig_vecs @ d_mtx @ self._eig_vecs.T
        x_w = self._white_mtx @ x_tensor

        return x_w.T
