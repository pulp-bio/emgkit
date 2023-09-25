"""Script that loads the synthetic isometric HDsEMG dataset proposed by Mohebian et al.
    in their work (https://doi.org/10.3389/fncom.2019.00014) and converts it into a
    non-isometric version.


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

import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from data_utils import load_synthetic_signal
from joblib import Parallel, delayed
from scipy import signal
from tqdm.auto import tqdm

import emgkit


def worker(
    gamma: float, gt_spikes_bin: np.ndarray, wfs: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Worker that computes the new EMG signal after stretching.

    Parameters
    ----------
    gamma : float
        Stretching factor.
    gt_spikes_bin : ndarray
        Binary representation of ground-truth spike trains with shape (waveform_len, n_mu).
    wfs : ndarray
        MUAPT waveforms with shape (n_channels, n_mu, waveform_len).
    t : ndarray
        Time index of the waveforms.

    Returns
    -------
    ndarray
        Signal with shape (n_ch,).
    """
    n_ch, n_mu, _ = wfs.shape

    def stretch_wfs(wfs_c: np.ndarray) -> np.ndarray:
        wfs_c_new_ = np.zeros_like(wfs_c)

        for m in range(n_mu):  # MUs
            wfs_c_new_[m] = np.interp(t / gamma, t, wfs_c[m]) / gamma
        return wfs_c_new_

    cur_emg = np.zeros(n_ch)
    for c in range(n_ch):  # channels
        wfs_c_new = stretch_wfs(wfs[c])
        wfs_c_new = np.flip(wfs_c_new, axis=1)
        cur_emg[c] = np.sum(wfs_c_new.T * gt_spikes_bin)
    return cur_emg


def main() -> None:
    if len(sys.argv) != 5:
        sys.exit(
            "Usage: python3 convert_to_non_iso DATA_PATH_IN DATA_PATH_OUT MVC CYCLE"
        )
    _, data_path_in, data_path_out, mvc, cycle = sys.argv
    mvc, cycle = int(mvc), int(cycle)

    min_gamma, max_gamma = 1.0, 1.2

    fs = 4096
    sig_len_s = 16.0

    wf_radius_ms = 25.0
    wf_radius = int(round(wf_radius_ms / 1000 * fs))

    stop_iso_s = 4.0
    stop_iso = int(round(stop_iso_s * fs))

    t_gamma = np.linspace(stop_iso_s, sig_len_s, int((sig_len_s - stop_iso_s) * fs))
    t_wf = np.arange(2 * wf_radius + 1) / fs

    n_cores = os.cpu_count()

    for s, (emg, gt_spikes_t, _) in enumerate(
        tqdm(load_synthetic_signal(data_path_in, mvc=mvc, apply_filter=False))
    ):
        # Compute waveforms
        wfs = emgkit.utils.compute_waveforms(emg, gt_spikes_t, wf_radius_ms, fs)

        # Dense representation of ground-truth spikes
        gt_spikes_bin = emgkit.utils.sparse_to_dense(
            gt_spikes_t, sig_len_s, fs
        ).to_numpy()
        gt_spikes_bin = np.pad(gt_spikes_bin, pad_width=((0, wf_radius), (0, 0)))

        # Partial worker
        part_worker = partial(worker, wfs=wfs, t=t_wf)

        gamma_cycle = signal.sawtooth(2 * np.pi * (t_gamma - stop_iso_s) / cycle, width=0.5)  # type: ignore
        gamma_cycle = (gamma_cycle + 1) / 2  # rescale to 0-1 range
        gamma_cycle = (
            gamma_cycle * (max_gamma - min_gamma) + min_gamma
        )  # rescale to gamma range

        # Keep first seconds isometric
        emg_new = np.zeros_like(emg.to_numpy())
        for c in range(wfs.shape[0]):  # channels
            for m in range(wfs.shape[1]):  # MUs
                emg_new[:stop_iso, c] += np.convolve(
                    gt_spikes_bin[: stop_iso + wf_radius, m], wfs[c, m], "same"
                )[:stop_iso]

        # Compute new signal in parallel
        results = Parallel(n_jobs=n_cores)(
            delayed(part_worker)(
                gamma,
                gt_spikes_bin[stop_iso + i - wf_radius : stop_iso + i + wf_radius + 1],
            )
            for i, gamma in enumerate(tqdm(gamma_cycle))
        )
        tmp = np.stack(results, axis=0)  # type: ignore
        emg_new[stop_iso:, :] = tmp

        # Convert both EMG and spikes to DataFrames
        emg_new_df = pd.DataFrame(
            data=emg_new,
            index=emg.index,
            columns=emg.columns,
        )
        gt_spikes_t_df = pd.DataFrame(
            data=[(mu, dt) for mu, spikes in gt_spikes_t.items() for dt in spikes],
            columns=["MU index", "Discharge time"],
        )
        # Save to H5 file
        out_file = os.path.join(data_path_out, f"MVC{mvc}S{s}C{cycle}.h5")
        emg_new_df.to_hdf(out_file, "EMG", format="table")
        gt_spikes_t_df.to_hdf(out_file, "Spikes", format="table", mode="a")


if __name__ == "__main__":
    main()
