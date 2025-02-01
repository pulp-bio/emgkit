"""
Class implementing the ZCA whitening algorithm.


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

from math import sqrt

import torch

from .._base import Signal, signal_to_tensor
from ._abc_whitening import WhiteningModel


class ZCAWhitening(WhiteningModel):
    """
    Class implementing ZCA whitening.

    Parameters
    ----------
    device : device or str, default="cpu"
        Torch device.

    Attributes
    ----------
    _device : device
        Torch device.
    _n_samp_seen : int
        Number of samples seen.
    _u : Tensor
        Left-singular vectors.
    _s : Tensor
        Singular values.
    _vt : Tensor
        Right-singular vectors.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self._device = torch.device(device) if isinstance(device, str) else device
        self._n_samp_seen = 0

        self._u: torch.Tensor = None  # type: ignore
        self._s: torch.Tensor = None  # type: ignore
        self._vt: torch.Tensor = None  # type: ignore

        self._mean_vec: torch.Tensor = None  # type: ignore
        self._white_mtx: torch.Tensor = None  # type: ignore
        self._cov_mtx: torch.Tensor = None  # type: ignore

    @property
    def mean_vec(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated mean vector."""
        return self._mean_vec

    @property
    def white_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated whitening matrix."""
        return self._white_mtx

    @property
    def cov_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the covariance matrix."""
        return self._cov_mtx

    def whiten_training(self, x: Signal) -> torch.Tensor:
        """
        Train the whitening model to whiten the given signal. If called multiple times,
        the model updates its internal parameters without forgetting the previous history.

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).

        Returns
        -------
        Tensor
            White signal with shape (n_samples, n_components).

        Raises
        ------
        TypeError
            If the input is neither an array, a DataFrame nor a Tensor.
        ValueError
            If the input is not 2D.
        """
        first_pass = self._n_samp_seen == 0

        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device).T
        n_samp = x_tensor.size(1)

        if first_pass:
            # Compute mean vector and center data
            self._mean_vec = x_tensor.mean(dim=1, keepdim=True)
            x_tensor -= self._mean_vec

            # Compute covariance matrix
            self._cov_mtx = x_tensor @ x_tensor.T / n_samp

            x_tensor_tmp = x_tensor
        else:
            # Compute weights for update
            n_samp_tot = self._n_samp_seen + n_samp
            w1 = self._n_samp_seen / n_samp_tot
            w2 = n_samp / n_samp_tot

            # Compute mean vector and center data
            mean_vec_new = x_tensor.mean(dim=1, keepdim=True)
            x_tensor -= mean_vec_new
            self._mean_vec = w1 * self._mean_vec + w2 * mean_vec_new

            # Compute covariance matrix
            cov_mtx = x_tensor @ x_tensor.T / n_samp
            self._cov_mtx = w1 * self._cov_mtx + w2 * cov_mtx

            # Compute mean correction
            mean_corr = sqrt(self._n_samp_seen * n_samp / n_samp_tot) * (
                self._mean_vec - mean_vec_new
            )

            # Compute new tensor
            x_tensor_tmp = torch.cat(
                (
                    x_tensor,  # new data
                    self._s * self._u @ self._vt,  # old data
                    mean_corr,
                ),
                dim=1,
            )

        # Update number of samples
        self._n_samp_seen += n_samp

        # SVD:
        # - the left-singular vectors of X are the eigenvectors of X @ X.T
        # - the singular values of X are the square root of the eigenvalues of X @ X.T
        # - the right-singular vectors of X are the eigenvectors of X.T @ X
        self._u, self._s, self._vt = torch.linalg.svd(x_tensor_tmp, full_matrices=False)
        self._u *= torch.sign(self._u[0])  # guarantee consistent sign

        # Whiten data
        eps = 1e-8
        d_mtx = torch.diag(1.0 / (self._s + eps)) * sqrt(self._n_samp_seen - 1)
        self._white_mtx = self._u @ d_mtx @ self._u.T
        x_w = self._white_mtx @ x_tensor

        return x_w.T

    def whiten_inference(self, x: Signal) -> torch.Tensor:
        """
        Whiten the given signal using the frozen whitening model.

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).

        Returns
        -------
        Tensor
            White signal with shape (n_samples, n_components).

        Raises
        ------
        TypeError
            If the input is neither an array, a DataFrame nor a Tensor.
        ValueError
            If the input is not 2D.
        """
        is_fit = self._mean_vec is not None and self._white_mtx is not None
        assert is_fit, "Fit the model first."

        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device).T

        # Center and whiten signal
        x_tensor -= self._mean_vec
        x_w = self._white_mtx @ x_tensor

        return x_w.T
