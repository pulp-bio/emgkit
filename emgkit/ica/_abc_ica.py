"""Interface for ICA-based algorithms.


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


class ICA(ABC):
    """Interface for models based on Independent Component Analysis."""

    @property
    @abstractmethod
    def mean_vec(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated mean vector."""

    @property
    @abstractmethod
    def sep_mtx(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated separation matrix."""

    @abstractmethod
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

    @abstractmethod
    def fit_transform(
        self,
        x: Signal,
        w_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fit the ICA model on the given signal and return the estimated sources.

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

    @abstractmethod
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
