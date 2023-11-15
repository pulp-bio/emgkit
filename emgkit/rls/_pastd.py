"""Class implementing the PASTd algorithm (https://doi.org/10.1109/78.365290) for whitening.


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
import torch

from .._base import Signal, signal_to_tensor


class PASTdW:
    """Class implementing the Projection Approximation Subspace Tracking (PAST) algorithm
    (https://doi.org/10.1109/78.365290) with deflation and Gram-Schmidt orthogonalization
    for whitening.

    Parameters
    ----------
    beta : float
        Forgetting factor:
        - a value of 1.0 corresponds to a growing sliding window (i.e., infinite memory);
        - a value < 1.0 corresponds to a sliding window with size 1 / (1 - beta) (i.e., finite memory).
    n_pcs : int
        Number of components to track.
    eig_vecs_init : ndarray or Tensor
        Initial matrix of eigenvectors with shape (n_channels, n_components).
    eig_vals_init : ndarray or Tensor
        Initial vector of eigenvalues with shape (n_components,).
    device : device or str or None, default=None
        Torch device.

    Attributes
    ----------
    _device : device or None
        Torch device.
    _n : int
        Number of iterations.
    """

    def __init__(
        self,
        beta: float,
        n_pcs: int,
        eig_vecs_init: np.ndarray | torch.Tensor,
        eig_vals_init: np.ndarray | torch.Tensor,
        device: torch.device | str | None = None,
    ) -> None:
        self._beta = beta
        self._n_pcs = n_pcs

        self._device = torch.device(device) if isinstance(device, str) else device

        self._eig_vecs = (
            torch.tensor(eig_vecs_init).to(self._device)
            if isinstance(eig_vecs_init, np.ndarray)
            else eig_vecs_init.clone().to(self._device)
        )
        self._eig_vals = (
            torch.tensor(eig_vals_init).to(self._device)
            if isinstance(eig_vals_init, np.ndarray)
            else eig_vals_init.clone().to(self._device)
        )

        self._n = 1

    @property
    def beta(self) -> float:
        """float: Property for getting the forgetting factor."""
        return self._beta

    @property
    def n_pcs(self) -> int:
        """int: Property for getting the number of components tracked."""
        return self._n_pcs

    @property
    def eig_vecs(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated matrix of eigenvectors."""
        return self._eig_vecs

    @property
    def eig_vals(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated vector of eigenvalues (scaled appropriately)."""
        alpha = 1 / self._n if self._beta == 1 else 1 - self._beta
        return self._eig_vals * alpha

    def process_sample(self, x: Signal) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single sample and adapt internal parameters.

        Parameters
        ----------
        x : Signal
            A signal with shape (1, n_channels) or (n_channels,).

        Returns
        -------
        Tensor
            Whitened sample with shape (1, n_components).
        Tensor
            Current eigenvalue with shape (n_components,).
        """
        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device, allow_1d=True).ravel()

        # Create residual signal
        x_res = x_tensor.clone()

        for i in range(self._n_pcs):
            # Apply 1D PAST
            z = self._eig_vecs[:, i] @ x_res
            self._eig_vals[i] = self._beta * self._eig_vals[i] + z**2
            err = x_res - self._eig_vecs[:, i] * z
            gain = z / self._eig_vals[i]
            self._eig_vecs[:, i] += err * gain

            # Apply Gram-Schmidt orthogonalization
            self._eig_vecs[:, i] -= (
                self._eig_vecs[:, i] @ self._eig_vecs[:, :i] @ self._eig_vecs[:, :i].T
            )
            self._eig_vecs[:, i] /= torch.linalg.norm(self._eig_vecs[:, i])

            # Subtract eigenvector contribution from residual
            x_res -= z * self._eig_vecs[:, i]

        # Compute whitened data
        alpha = 1 / self._n if self._beta == 1 else 1 - self._beta
        eig_vals = self._eig_vals * alpha
        d_mtx = torch.diag(1 / torch.sqrt(eig_vals))
        white_mtx = d_mtx @ self._eig_vecs.T
        x_w = torch.unsqueeze(white_mtx @ x_tensor, dim=0)

        self._n += 1

        return x_w, eig_vals
