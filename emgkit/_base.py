"""This module contains the basic type for signals.


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
import pandas as pd
import torch

Signal = np.ndarray | pd.DataFrame | pd.Series | torch.Tensor


def signal_to_tensor(
    x: Signal, device: torch.device | None = None, allow_1d: bool = False
) -> torch.Tensor:
    """Convert the signal to a Tensor.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    device : device or None, default=None
        Torch device.
    allow_1d : bool, default=True
        Whether to allow 1D signals.

    Returns
    -------
    Tensor
        The corresponding Tensor.

    Raises
    ------
    TypeError
        If the input is neither an array, a DataFrame/Series nor a Tensor.
    ValueError
        If the input is not 2D (or 1D if allow_1d is True).
    """
    # Convert input to Tensor
    if isinstance(x, np.ndarray):
        x_t = torch.from_numpy(x).to(device)
    elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x_t = torch.from_numpy(x.to_numpy()).to(device)
    elif isinstance(x, torch.Tensor):
        x_t = x.to(device)
    else:
        raise TypeError(
            "The input is neither an array, a DataFrame/Series nor a Tensor."
        )
    # Check shape
    if allow_1d and len(x_t.size()) == 1:
        x_t = torch.unsqueeze(x_t, dim=1)  # un-squeeze along channel dimension
    if len(x_t.size()) != 2:
        raise ValueError("The input is not 2D.")

    return x_t


def signal_to_array(x: Signal, allow_1d: bool = False) -> np.ndarray:
    """Convert the signal to an array.

    Parameters
    ----------
    x : Signal
        A signal with shape (n_samples, n_channels).
    allow_1d : bool, default=True
        Whether to allow 1D signals.

    Returns
    -------
    ndarray
        The corresponding array.

    Raises
    ------
    TypeError
        If the input is neither an array, a DataFrame/Series nor a Tensor.
    ValueError
        If the input is not 2D (or 1D if allow_1d is True).
    """
    # Convert input to array
    if isinstance(x, np.ndarray):
        x_a = x
    elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x_a = x.to_numpy()
    elif isinstance(x, torch.Tensor):
        x_a = x.cpu().numpy()
    else:
        raise TypeError(
            "The input is neither an array, a DataFrame/Series nor a Tensor."
        )
    # Check shape
    if allow_1d and len(x_a.shape) == 1:
        x_a = np.expand_dims(x_a, axis=1)  # un-squeeze along channel dimension
    if len(x_a.shape) != 2:
        raise ValueError("The input is not 2D.")

    return x_a
