"""Function performing the extension step of an EMG signal.


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

from .._base import Signal, signal_to_array


def extend_signal(x: Signal, f_ext: int = 1) -> np.ndarray:
    """Extend signal with delayed replicas by a given extension factor.

    Parameters
    ----------
    x : Signal
        A signal with shape:
        - (n_samples,);
        - (n_samples, n_channels).
    f_ext : int, default=1
        Extension factor.

    Returns
    -------
    ndarray
        Extended signal with shape (f_ext * n_channels, n_samples).

    Raises
    ------
    TypeError
        If the input is neither an array, a DataFrame/Series nor a Tensor.
    ValueError
        If the input is neither 2D nor 1D.
    """

    # Convert input to array
    x_a = signal_to_array(x, allow_1d=True)

    n_samp, n_ch = x_a.shape
    n_ch_ext = f_ext * n_ch

    x_ext = np.zeros(shape=(n_samp, n_ch_ext), dtype=x_a.dtype)
    for i in range(f_ext):
        x_ext[i:, i * n_ch : (i + 1) * n_ch] = x_a[: n_samp - i]
    return x_ext
