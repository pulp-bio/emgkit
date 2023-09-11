"""Function to load the sEMG signal acquired from our ESB armband.


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

import struct

import numpy as np
import pandas as pd


def load_signal_esb(data_path: str) -> pd.DataFrame:
    """Load sEMG data acquired using our ESB armband.

    Parameters
    ----------
    data_path : str
        Path to the dataset root folder.

    Returns
    -------
    DataFrame
        The sEMG signal with shape (n_samples, n_channels).
    """
    fs = 4000

    # Read file
    with open(data_path, "rb") as f:
        n_ch = struct.unpack("<I", f.read(4))[0]
        b_data = bytes(f.read())
    data = np.frombuffer(b_data, dtype="float32").reshape(-1, n_ch)
    n_ch -= 1
    n_samp = data.shape[0]

    # Pack in DataFrame
    emg = pd.DataFrame(
        data=data,
        index=np.arange(n_samp) / fs,
        columns=[f"Ch{i}" for i in range(n_ch)] + ["Trigger"],
    )
    emg["Trigger"] = emg["Trigger"].astype("int32")

    return emg
