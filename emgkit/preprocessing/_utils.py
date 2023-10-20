"""Internal utilities for preprocessing.


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

import warnings

import torch


def eigendecomposition(x_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform eigendecomposition of a given Tensor.

    Parameters
    ----------
    x_tensor: Tensor
        Input Tensor with shape (n_channels, n_samples).

    Returns
    -------
    Tensor:
        2D Tensor of eigenvectors sorted by the corresponding eigenvalue in descending order.
    Tensor:
        1D Tensor of eigenvalues sorted in descending order.
    """
    n_samp = x_tensor.size(1)
    cov_mtx = x_tensor @ x_tensor.T / n_samp
    d, e = torch.linalg.eigh(cov_mtx)

    # Improve numerical stability
    eps = torch.finfo(d.dtype).eps
    degenerate_idx = torch.lt(d, eps).nonzero()
    if torch.any(degenerate_idx):
        warnings.warn(f"Some eigenvalues are smaller than epsilon ({eps:.3e}).")
    d[degenerate_idx] = eps

    sort_idx = torch.argsort(d, descending=True)
    d, e = d[sort_idx], e[:, sort_idx]

    return e, d
