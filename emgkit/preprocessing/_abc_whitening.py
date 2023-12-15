"""Interface for whitening algorithms.


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

from abc import ABC, abstractmethod

import torch

from .._base import Signal


class WhiteningModel(ABC):
    """Interface for performing whitening."""

    @property
    @abstractmethod
    def mean_vec(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated mean vector."""

    @property
    @abstractmethod
    def white_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated whitening matrix."""

    @property
    @abstractmethod
    def cov_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the covariance matrix."""

    @abstractmethod
    def whiten_training(self, x: Signal) -> torch.Tensor:
        """Train the whitening model to whiten the given signal. If called multiple times,
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
            If the input is neither an array, a DataFrame/Series nor a Tensor.
        ValueError
            If the input is not 2D.
        """

    @abstractmethod
    def whiten_inference(self, x: Signal) -> torch.Tensor:
        """Whiten the given signal using the frozen whitening model.

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
            If the input is neither an array, a DataFrame/Series nor a Tensor.
        ValueError
            If the input is not 2D.
        """
