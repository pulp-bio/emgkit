"""Class implementing Natural Gradient-based Recursive Least Squares algorithms for BSS.


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


def _g_super(x: torch.Tensor) -> torch.Tensor:
    """Optimization function for super-Gaussian sources."""
    return x - torch.tanh(x)


def _g_sub(x: torch.Tensor) -> torch.Tensor:
    """Optimization function for sub-Gaussian sources."""
    return torch.tanh(x)


def _sym_orth_ap(w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Approximated symmetric orthogonalization procedure."""
    max_iter = w.size(1)

    for _ in range(max_iter):
        w /= torch.sqrt(torch.linalg.norm(w @ r @ w.T, ord=1))
        w = 1.5 * w - 0.5 * w @ r @ w.T @ w

    return w


class NatGradRLSPreWhite:
    """Class implementing the Natural Gradient Recursive Least Squares (NG-RLS) algorithm
    (https://doi.org/10.1109/LSP.2002.806047) for BSS. This version requires data to be pre-whitened.

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
    w_mtx_init : ndarray or Tensor
        Initial W matrix with shape (n_components, n_channels).
    p_mtx_init : ndarray or Tensor
        Initial P matrix with shape (n_components, n_components).
    device : device or str or None, default=None
        Torch device.

    Attributes
    ----------
    _g_func : Callable[[Tensor], Tensor]
        Optimization function.
    _device : device or None
        Torch device.
    _p_mtx : Tensor
        P matrix with shape (n_components, n_components).
    """

    def __init__(
        self,
        beta: float,
        n_ics: int,
        kurtosis: str,
        w_mtx_init: np.ndarray | torch.Tensor,
        p_mtx_init: np.ndarray | torch.Tensor,
        device: torch.device | str | None = None,
    ) -> None:
        assert kurtosis in (
            "super",
            "sub",
        ), '"kurtosis" must be either "super" or "sub".'

        self._beta = beta
        self._n_ics = n_ics

        g_dict = {
            "super": _g_super,
            "sub": _g_sub,
        }
        self._g_func = g_dict[kurtosis]

        self._device = torch.device(device) if isinstance(device, str) else device

        self._w_mtx = (
            torch.tensor(w_mtx_init).to(self._device)
            if isinstance(w_mtx_init, np.ndarray)
            else w_mtx_init.clone().to(self._device)
        )
        self._p_mtx = (
            torch.tensor(p_mtx_init).to(self._device)
            if isinstance(p_mtx_init, np.ndarray)
            else p_mtx_init.clone().to(self._device)
        )

    @property
    def beta(self) -> float:
        """float: Property for getting the forgetting factor."""
        return self._beta

    @property
    def n_ics(self) -> int:
        """int: Property for getting the number of components tracked."""
        return self._n_ics

    @property
    def w_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the W matrix."""
        return self._w_mtx

    def process_sample(self, x: Signal) -> torch.Tensor:
        """Process a single sample from a whitened signal and adapt internal parameters.

        Parameters
        ----------
        x : Signal
            A sample from a whitened signal with shape (1, n_channels) or (n_channels,).

        Returns
        -------
        Tensor
            Separated sample with shape (1, n_components).
        """
        # Convert input to Tensor
        x_tensor = signal_to_tensor(x, self._device, allow_1d=True).ravel()[:, None]

        y = self._w_mtx @ x_tensor
        z = self._g_func(y)
        q_mtx = self._p_mtx / (self._beta + z.T @ self._p_mtx @ y)
        self._p_mtx = (self._p_mtx - q_mtx @ y @ z.T @ self._p_mtx) / self._beta
        self._w_mtx += self._p_mtx @ z @ x_tensor.T - q_mtx @ y @ z.T @ self._w_mtx

        return (self._w_mtx @ x_tensor).T


class NatGradRLS:
    """Class implementing the Natural Gradient Recursive Least Squares (NG-RLS) algorithm
    (https://doi.org/10.1109/TCSI.2005.858489) for BSS without pre-whitening.

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
    w_mtx_init : ndarray or Tensor
        Initial W matrix with shape (n_components, n_channels).
    p_mtx_init : ndarray or Tensor
        Initial P matrix with shape (n_components, n_components).
    cov_mtx_init : ndarray or Tensor or None, default=None
        Initial covariance matrix with shape (n_channels, n_channels).
    device : device or str or None, default=None
        Torch device.

    Attributes
    ----------
    _g_func : Callable[[Tensor], Tensor]
        Optimization function.
    _device : device or None
        Torch device.
    _p_mtx : Tensor
        P matrix with shape (n_components, n_components).
    _cov_mtx : Tensor
        Covariance matrix.
    """

    def __init__(
        self,
        beta: float,
        n_ics: int,
        kurtosis: str,
        w_mtx_init: np.ndarray | torch.Tensor,
        p_mtx_init: np.ndarray | torch.Tensor,
        cov_mtx_init: np.ndarray | torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        assert kurtosis in (
            "super",
            "sub",
        ), '"kurtosis" must be either "super" or "sub".'

        self._beta = beta
        self._n_ics = n_ics

        g_dict = {
            "super": _g_super,
            "sub": _g_sub,
        }
        self._g_func = g_dict[kurtosis]

        self._device = torch.device(device) if isinstance(device, str) else device

        self._w_mtx = (
            torch.tensor(w_mtx_init).to(self._device)
            if isinstance(w_mtx_init, np.ndarray)
            else w_mtx_init.clone().to(self._device)
        )
        self._p_mtx = (
            torch.tensor(p_mtx_init).to(self._device)
            if isinstance(p_mtx_init, np.ndarray)
            else p_mtx_init.clone().to(self._device)
        )

        if cov_mtx_init is not None:
            self._cov_mtx = (
                torch.tensor(cov_mtx_init).to(self._device)
                if isinstance(cov_mtx_init, np.ndarray)
                else cov_mtx_init.clone().to(self._device)
            )

    @property
    def beta(self) -> float:
        """float: Property for getting the forgetting factor."""
        return self._beta

    @property
    def n_ics(self) -> int:
        """int: Property for getting the number of components tracked."""
        return self._n_ics

    @property
    def w_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the W matrix."""
        return self._w_mtx

    def process_sample(self, x: Signal, approx=False) -> torch.Tensor:
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
        n_ch = x_tensor.shape[0]

        y = self._w_mtx @ x_tensor
        z = self._g_func(y)
        q_mtx = self._p_mtx / (self._beta + z.T @ self._p_mtx @ y)
        self._p_mtx = (self._p_mtx - q_mtx @ y @ z.T @ self._p_mtx) / self._beta
        self._w_mtx += (self._p_mtx @ z @ y.T - q_mtx @ y @ z.T) @ self._w_mtx

        # Update covariance matrix
        if not hasattr(self, "_cov_mtx"):
            self._cov_mtx = torch.zeros(
                n_ch, n_ch, dtype=torch.float32, device=self._device
            )
        self._cov_mtx = (
            self._beta * self._cov_mtx + (1 - self._beta) * x_tensor @ x_tensor.T
        )

        # Orthogonalization
        self._w_mtx = _sym_orth_ap(self._w_mtx, self._cov_mtx)

        return (self._w_mtx @ x_tensor).T
