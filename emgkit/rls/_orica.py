"""Class implementing the ORICA algorithm.


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
from ..preprocessing import ZCAWhitening


def _g_super(x: torch.Tensor) -> torch.Tensor:
    """Optimization function for super-Gaussian sources."""
    return -2 * torch.tanh(x)


def _g_sub(x: torch.Tensor) -> torch.Tensor:
    """Optimization function for sub-Gaussian sources."""
    return torch.tanh(x)


def _sym_orth_ap(w: torch.Tensor) -> torch.Tensor:
    """Approximated symmetric orthogonalization procedure."""
    max_iter = w.size(1)

    for _ in range(max_iter):
        w /= torch.sqrt(torch.linalg.norm(w, ord=1))
        w = 1.5 * w - 0.5 * w @ w.T @ w

    return w


class ORICA:
    """Class implementing the ORICA algorithm.

    Parameters
    ----------
    beta : float
        Forgetting factor:
        - a value of 1.0 corresponds to a growing sliding window (i.e., infinite memory);
        - a value < 1.0 corresponds to a sliding window with size 1 / (1 - beta) (i.e., finite memory).
    n_ics : int
        Number of components to track.
    kurtosis: {"super", "sub"}
        String representing the kurtosis of the source signals (i.e., either "super" or "sub").
    u_mtx_init : ndarray or Tensor
        Initial whitening matrix with shape (n_pcs, n_components).
    w_mtx_init : ndarray or Tensor
        Initial separation matrix with shape (n_ics, n_pcs).
    device : device or str, default="cpu"
        Torch device.

    Attributes
    ----------
    _g_func : Callable[[Tensor], Tensor]
        Optimization function.
    _device : device
        Torch device.
    _n : int
        Number of iterations.
    """

    def __init__(
        self,
        beta: float,
        n_ics: int,
        u_mtx_init: np.ndarray | torch.Tensor,
        w_mtx_init: np.ndarray | torch.Tensor,
        device: torch.device | str = "cpu",
    ) -> None:
        self._beta0 = beta
        self._beta = beta
        self._n_ics = n_ics

        self._g_func = _g_super

        self._device = torch.device(device) if isinstance(device, str) else device

        self._u_mtx = (
            torch.tensor(u_mtx_init).to(self._device)
            if isinstance(u_mtx_init, np.ndarray)
            else u_mtx_init.clone().to(self._device)
        )
        self._w_mtx = (
            torch.tensor(w_mtx_init).to(self._device)
            if isinstance(w_mtx_init, np.ndarray)
            else w_mtx_init.clone().to(self._device)
        )

        self._n = 1

    @property
    def beta(self) -> float:
        """float: Property for getting the forgetting factor."""
        return self._beta

    @property
    def n_ics(self) -> int:
        """int: Property for getting the number of components tracked."""
        return self._n_ics

    @property
    def u_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the whitening matrix."""
        return self._u_mtx

    @property
    def w_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the separation matrix."""
        return self._w_mtx

    def process_sample(self, x: Signal) -> torch.Tensor:
        """Process a single sample and adapt internal parameters.

        Parameters
        ----------
        x : Signal
            A sample with shape (1, n_channels) or (n_channels,).

        Returns
        -------
        Tensor
            Separated sample with shape (1, n_components).
        """
        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device, allow_1d=True).ravel()[:, None]

        alpha = self._beta / (1 - self._beta)

        # RLS whitening
        v = self._u_mtx @ x_tensor
        self._u_mtx = (
            1 / self._beta * (self._u_mtx - v @ v.T / (alpha * v.T @ v) @ self._u_mtx)
        )
        v = self._u_mtx @ x_tensor

        self._beta = self._beta0 / self._n**0.6
        self._n += 1

        eye = torch.eye(v.size(0), dtype=v.dtype, device=v.device)
        return ((eye - v @ v.T) ** 2).mean() ** 0.5

        # RLS BSS
        y = self._w_mtx @ v
        z = self._g_func(y)
        self._w_mtx = (
            1 / self._beta * (self._w_mtx - y @ z.T / (alpha * z.T @ y) @ self._w_mtx)
        )

        # Orthogonalization
        # self._w_mtx = _sym_orth_ap(self._w_mtx)

        return (self._w_mtx @ self._u_mtx @ x_tensor).T

    def process_window(self, x: Signal) -> torch.Tensor:
        """Process a window and adapt internal parameters.

        Parameters
        ----------
        x : Signal
            A sample with shape (n_samples, n_channels).

        Returns
        -------
        Tensor
            Separated sample with shape (n_samples, n_components).
        """
        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device).T
        n_ch, n_samp = x_tensor.size()

        alpha = self._beta / (1 - self._beta)

        v = ZCAWhitening(device=self._device).whiten_training(x_tensor.T).T

        # RLS whitening
        # v = self._u_mtx @ x_tensor
        # num = v @ v.T / (n_samp - 1)
        # den = alpha * (v * v).sum().item() / (n_samp - 1)
        # self._u_mtx = 1 / self._beta * (self._u_mtx - num / den @ self._u_mtx)
        # v = self._u_mtx @ x_tensor

        # self._beta = 1 - (1 - self._beta0) / self._n**0.6

        eye = torch.eye(v.size(0), dtype=v.dtype, device=v.device)
        return ((eye - v @ v.T / (n_samp - 1)) ** 2).mean() ** 0.5

        # RLS BSS
        y = self._w_mtx @ v
        z = self._g_func(y)
        num = y @ z.T / (n_samp - 1)
        den = alpha * torch.trace(y.T @ z).item() / (n_samp - 1)
        self._w_mtx = 1 / self._beta * (self._w_mtx - num / den @ self._w_mtx)

        self._n += n_samp

        # Orthogonalization
        self._w_mtx = _sym_orth_ap(self._w_mtx)

        return (self._w_mtx @ v).T
