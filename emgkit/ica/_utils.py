"""Internal utility functins for ICA.


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

import torch


def sym_orth(w: torch.Tensor) -> torch.Tensor:
    """Helper function to perform symmetric orthogonalization."""
    eig_vals, eig_vecs = torch.linalg.eigh(w @ w.T)

    # Improve numerical stability
    eig_vals = torch.clip(eig_vals, min=torch.finfo(w.dtype).tiny)

    d_mtx = torch.diag(1.0 / torch.sqrt(eig_vals))
    return eig_vecs @ d_mtx @ eig_vecs.T @ w


class ConvergenceWarning(Warning):
    """Warning related to an algorithm not converging."""
