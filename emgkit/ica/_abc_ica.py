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
from ..preprocessing import WhiteningModel


class ICA(ABC):
    """Interface for models based on Independent Component Analysis."""

    @property
    @abstractmethod
    def sep_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated separation matrix."""

    @property
    @abstractmethod
    def whiten_model(self) -> WhiteningModel | None:
        """WhiteningModel or None: Property for getting the whitening model."""

    @abstractmethod
    def decompose_training(self, x: Signal) -> torch.Tensor:
        """Train the ICA model to decompose the given signal into independent components (ICs).

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).

        Returns
        -------
        Tensor
            Estimated ICs with shape (n_samples, n_components).

        Warns
        -----
        ConvergenceWarning
            The algorithm didn't converge.
        """

    @abstractmethod
    def decompose_inference(self, x: Signal) -> torch.Tensor:
        """Decompose the given signal into independent components (ICs) using the frozen ICA model.

        Parameters
        ----------
        x : Signal
            A signal with shape (n_samples, n_channels).

        Returns
        -------
        Tensor
            Estimated ICs with shape (n_samples, n_components).
        """
