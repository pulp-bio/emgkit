"""Functions for visualizations.


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

from collections import Counter
from collections.abc import Collection
from itertools import groupby

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import colormaps as cm
from matplotlib import patches as mpl_patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from .._base import Signal
from ..spike_stats import instantaneous_discharge_rate, smoothed_discharge_rate
from ..utils import sparse_to_dense

# Set Seaborn style
sns.set_theme(style="whitegrid")


def _plot_signal(
    s_df: pd.DataFrame,
    labels: pd.Series | None,
    title: str | None = None,
    x_label: str = "Time [s]",
    y_label: str = "Amplitude [a.u.]",
    fig_size: tuple[int, int] | None = None,
) -> None:
    """Helper function to plot a signal with multiple channels, each in a different subplot."""
    # Create figure with subplots and shared X axis
    n_cols = 1
    n_rows = s_df.shape[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex="all",
        squeeze=False,
        figsize=fig_size,
        layout="constrained",
    )
    axes = [ax for nested_ax in axes for ax in nested_ax]  # flatten axes
    # Set title and label of X and Y axes
    if title is not None:
        fig.suptitle(title, fontsize="xx-large")
    fig.supxlabel(x_label)
    fig.supylabel(y_label)

    # Plot signal
    if labels is not None:
        # Get label intervals
        labels_intervals = []
        labels_tmp = [
            list(group)
            for _, group in groupby(
                labels.reset_index().to_numpy().tolist(), key=lambda t: t[1]
            )
        ]
        for cur_label in labels_tmp:
            cur_label_start, cur_label_name = cur_label[0]
            cur_label_stop = cur_label[-1][0]
            labels_intervals.append((cur_label_name, cur_label_start, cur_label_stop))
        # Get set of unique labels
        label_set = set(map(lambda t: t[0], labels_intervals))
        # Create dictionary label -> color
        cmap = cm["tab20"].resampled(len(label_set))
        color_dict = {lab: cmap(i) for i, lab in enumerate(label_set)}
        for i, ch_i in enumerate(s_df):
            for label, idx_from, idx_to in labels_intervals:
                axes[i].plot(
                    s_df[ch_i].loc[idx_from:idx_to],
                    color=color_dict[label],
                )
        # Add legend
        fig.legend(
            handles=[
                mpl_patches.Patch(color=c, label=lab) for lab, c in color_dict.items()
            ],
            loc="center right",
        )
    else:
        for i, ch_i in enumerate(s_df):
            axes[i].plot(s_df[ch_i])


def _plot_signal_heatmap(
    s_df: pd.DataFrame,
    labels: pd.Series | None,
    title: str | None = None,
    x_label: str = "Time [s]",
    y_label: str = "Channels",
    fig_size: tuple[int, int] | None = None,
    resolution: int | None = None,
) -> None:
    """Helper function to plot a signal with multiple channels as a compact heatmap."""
    cmap = "icefire" if (s_df.min() < 0).any() else "magma"

    # Rolling mean
    if resolution is not None:
        s_df = s_df.rolling(resolution, step=resolution).mean().dropna()

    if labels is not None:
        # Rolling mode on labels
        if resolution is not None:
            l2i = {lab: i for i, lab in enumerate(labels.unique())}
            i2l = {i: lab for i, lab in enumerate(labels.unique())}
            labels = (
                labels.map(l2i)
                .rolling(resolution, step=resolution)
                .apply(lambda x: x.iloc[-1])
                .dropna()
                .map(i2l)
            )

        # Create figure with 3 subplots (one for the heatmap, one for the color bar and one for the labels)
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=fig_size,
            layout="constrained",
            gridspec_kw={"width_ratios": (31, 1), "height_ratios": (1, 31)},
        )
        # Remove extra subplot
        fig.delaxes(axes[0, 1])
        # Set title and label of X axis
        if title is not None:
            fig.suptitle(title, fontsize="xx-large")
        fig.supxlabel(x_label)

        # Create dictionary label -> color
        label_cmap = cm["viridis"].resampled(labels.nunique())
        color_dict = {lab: label_cmap(i) for i, lab in enumerate(labels.unique())}
        # Add legend
        fig.legend(
            handles=[
                mpl_patches.Patch(color=c, label=lab) for lab, c in color_dict.items()
            ],
            loc="upper right",
        )

        # Scatter plot for labels
        axes[0, 0].scatter(
            x=np.arange(labels.size),
            y=np.zeros(shape=labels.size, dtype="uint8"),
            c=labels.map(color_dict),
            marker=".",
        )
        axes[0, 0].set_xbound(0, labels.size)
        axes[0, 0].set_axis_off()

        # Plot heatmap
        sns.heatmap(
            s_df.T,
            cmap=cmap,
            robust=True,
            cbar_ax=axes[1, 1],
            xticklabels=axes[0, 0].get_xticks(),
            ax=axes[1, 0],
        )
        axes[1, 0].set_ylabel(y_label)
    else:
        # Create figure with 2 subplots (one for the heatmap, the other for the color bar)
        fig, axes = plt.subplots(
            ncols=2,
            figsize=fig_size,
            layout="constrained",
            gridspec_kw={"width_ratios": (31, 1)},
        )
        # Set title and label of X and Y axes
        if title is not None:
            fig.suptitle(title, fontsize="xx-large")
        fig.supxlabel(x_label)
        fig.supylabel(y_label)

        # Plot heatmap
        s_df1 = s_df
        s_df1.index = s_df.index.map(lambda t: f"{t:.3f}")
        sns.heatmap(
            s_df1.T,
            cmap=cmap,
            robust=True,
            cbar_ax=axes[1],
            ax=axes[0],
        )


def plot_signal(
    s: Signal,
    fs: float = 1.0,
    labels: np.ndarray | pd.Series | None = None,
    as_heatmap: bool = False,
    title: str | None = None,
    x_label: str = "Time [s]",
    y_label: str = "Amplitude [a.u.]",
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
    resolution: int | None = None,
) -> None:
    """Plot a signal with multiple channels.

    Parameters
    ----------
    s : Signal
        A signal with shape (n_samples, n_channels).
    fs : float, default=1.0
        Sampling frequency of the signal (relevant if s is a NumPy array).
    labels : ndarray or Series or None, default=None
        NumPy array or Series containing a label for each sample.
    as_heatmap : bool, default=False
        Whether to plot a compact heatmap or the complete signal.
    title : str or None, default=None
        Title of the plot.
    x_label : str, default="Time [s]"
        Label for X axis.
    y_label : str, default="Amplitude [a.u.]"
        Label for Y axis.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    resolution : int or None, default=None
        Resolution of the heatmap (relevant only for compact style).
    """
    # Convert signal to DataFrame
    if isinstance(s, pd.DataFrame):
        s_df = s
    elif isinstance(s, pd.Series):
        s_df = s.to_frame()
    else:
        s_array = s.cpu().numpy() if isinstance(s, torch.Tensor) else s
        if len(s_array.shape) == 1:
            s_array = s_array.reshape(-1, 1)
        s_df = pd.DataFrame(s_array, index=np.arange(s_array.shape[0]) / fs)

    # Convert labels to Series
    labels_s = (
        pd.Series(labels, index=np.arange(labels.size) / fs)
        if isinstance(labels, np.ndarray)
        else labels
    )

    # Plot signal
    if as_heatmap:
        _plot_signal_heatmap(
            s_df, labels_s, title, x_label, y_label, fig_size, resolution
        )
    else:
        _plot_signal(s_df, labels_s, title, x_label, y_label, fig_size)
    args = [s_df, labels, title, x_label, y_label, fig_size]
    if as_heatmap:
        args.append(resolution)

    # Show or save plot
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_waveforms(
    wfs: np.ndarray,
    fs: float,
    n_cols: int = 10,
    y_label: str = "Amplitude [a.u.]",
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Function to plot MUAP waveforms.

    Parameters
    ----------
    wfs : ndarray
        MUAP waveforms with shape (n_channels, n_mu, waveform_len).
    fs : float
        Sampling frequency of the signal.
    n_cols : int, default=10
        Number of columns for subplots.
    y_label : str, default="Amplitude [a.u.]"
        Label for Y axis.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    n_ch = wfs.shape[0]
    assert (
        n_ch % n_cols == 0
    ), "The number of channels must be divisible for the number of columns."
    n_rows = n_ch // n_cols
    t = np.arange(wfs.shape[2]) * 1000 / fs

    f, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex="all",
        sharey="all",
        squeeze=False,
        figsize=fig_size,
        layout="constrained",
    )

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            axes[i, j].set_title(f"Ch{idx}")
            axes[i, j].plot(t, wfs[idx].T)

    f.suptitle("MUAP waveforms")
    f.supxlabel("Time [ms]")
    f.supylabel(y_label)

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_correlation(
    s: Signal,
    write_annotations: bool = False,
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Plot the correlation matrix between the channels of a given signal.

    Parameters
    ----------
    s : Signal
        A signal with shape (n_samples, n_channels).
    write_annotations : bool, default=False
        Whether to write annotations inside the correlation matrix or not.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    # Convert to DataFrame
    if isinstance(s, pd.DataFrame):
        s_df = s
    else:
        s_array = s.cpu().numpy() if isinstance(s, torch.Tensor) else s
        s_df = pd.DataFrame(s_array)

    # Compute correlation and plot heatmap
    corr = s_df.corr()
    _, ax = plt.subplots(figsize=fig_size, layout="constrained")
    sns.heatmap(
        corr,
        vmax=1.0,
        vmin=-1.0,
        cmap="icefire",
        annot=write_annotations,
        square=True,
        ax=ax,
    )

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_ic_spikes(
    ics: pd.DataFrame,
    spikes_t: dict[str, np.ndarray],
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Plot the given ICs and spikes.

    Parameters
    ----------
    ics : DataFrame
        A DataFrame with shape (n_samples, n_mu) containing the components estimated by ICA.
    spikes_t : dict of {str : ndarray}
        Dictionary containing the discharge times for each MU.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    assert ics.shape[1] == len(
        spikes_t
    ), "The number of ICs must match the number of spike trains."

    f, axes = plt.subplots(
        nrows=len(spikes_t),
        sharex="all",
        squeeze=False,
        figsize=fig_size,
        layout="constrained",
    )
    axes = [ax for nested_ax in axes for ax in nested_ax]  # flatten axes
    f.suptitle("ICs spike trains")
    f.supxlabel("Time [s]")
    f.supylabel("Amplitude [a.u.]")

    for i, mu in enumerate(spikes_t):
        axes[i].plot(ics[mu])
        axes[i].plot(spikes_t[mu], ics[mu].loc[spikes_t[mu]], "x")

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def raster_plot(
    spikes_t: dict[str, np.ndarray],
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Plot a raster plot of the given discharges.

    Parameters
    ----------
    spikes_t : dict of {str : ndarray}
        Dictionary containing the discharge times for each MU.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    f, ax = plt.subplots(figsize=fig_size, layout="constrained")
    f.suptitle("Raster plot")
    f.supxlabel("Time [s]")
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])

    for i, mu in enumerate(spikes_t.keys()):
        ax.scatter(
            x=spikes_t[mu],
            y=[len(spikes_t) - i] * spikes_t[mu].size,
            marker="|",
        )

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_discharges(
    spikes_t: dict[str, np.ndarray],
    sig_len_s: float,
    fs: float,
    win_len_s: float = 1.0,
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Plot the discharge rate of each MU.

    Parameters
    ----------
    spikes_t : dict of {str : ndarray}
        Dictionary containing the discharge times for each MU.
    sig_len_s : float
        Length of the signal (in seconds).
    fs : float
        Sampling frequency.
    win_len_s : float, default=1.0
        Size (in seconds) of the Hanning window.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    f, axes = plt.subplots(
        nrows=len(spikes_t),
        sharex="all",
        squeeze=False,
        figsize=fig_size,
        layout="constrained",
    )
    axes = [ax for nested_ax in axes for ax in nested_ax]  # flatten axes
    f.suptitle("Discharge rate")
    f.supxlabel("Time [s]")
    f.supylabel("Discharge rate [spike/s]")

    # Dense representation
    spikes_bin = sparse_to_dense(spikes_t, sig_len_s, fs)

    for i, mu in enumerate(spikes_t):
        smooth_dr = smoothed_discharge_rate(spikes_bin[mu], fs, win_len_s=win_len_s)
        inst_dr = instantaneous_discharge_rate(spikes_t[mu])
        axes[i].set_title(mu)
        axes[i].plot(smooth_dr, label="Smooth")
        axes[i].plot(inst_dr, ".", label="Instantaneous")
    handles, labels = axes[-1].get_legend_handles_labels()
    f.legend(handles, labels, loc="center right")

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def _data_distribution_helper(
    y: np.ndarray, label_dict: dict[str, int], ax: Axes
) -> None:
    """Helper function to plot the distribution of the labels of a single dataset."""
    # Count labels
    y_count = Counter(y)

    g_range = list(label_dict.values())
    ax.bar(
        x=g_range,
        height=[y_count[g] for g in g_range],
    )
    ax.set_xticks(g_range)
    ax.set_xticklabels(label_dict.keys(), rotation=45)


def plot_data_distribution(
    y: np.ndarray | Collection[np.ndarray],
    label_dict: dict[str, int],
    title: str | Collection[str] | None = None,
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Plot the distribution of the labels of a given dataset (or list of datasets).

    Parameters
    ----------
    y : ndarray or list of ndarray
        An array (or a list of arrays) containing the labels of a dataset.
    label_dict : dict of {str, int}
        Dictionary mapping string labels to the respective integer labels.
    title : str or list of str or None, default=None
        String (or list of strings) representing the title(s) of the plot(s).
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    # Check for single or multiple plots
    if isinstance(y, np.ndarray):  # single plot
        assert (
            isinstance(title, str) or title is None
        ), "'y' is a single array, thus 'title' should be single as well."

        # Create figure
        _, ax = plt.subplots(figsize=fig_size, layout="constrained")

        # Plot bar plot
        _data_distribution_helper(y, label_dict, ax)

        # Set title
        if title is not None:
            ax.set_title(title)
    elif isinstance(y, Collection):  # multiple plots
        assert (
            isinstance(title, Collection) and len(y) == len(title)
        ) or title is None, (
            "'y' is a list of arrays, thus 'title' (if provided) should be a list as well, "
            "with the same length as 'y'."
        )

        # Compute number of rows
        n_plots = len(y)
        n_cols = 2
        mod = n_plots % n_cols
        n_rows = n_plots // n_cols if mod == 0 else n_plots // n_cols + mod
        # Create figure with subplots and shared x-axis
        _, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex="all",
            sharey="all",
            squeeze=False,
            figsize=fig_size,
            layout="constrained",
        )
        axes = [ax for nested_ax in axes for ax in nested_ax]  # flatten axes

        # Plot barplots
        opt_title_list = [None] * len(y) if title is None else title
        for y, t, a in zip(y, opt_title_list, axes):
            _data_distribution_helper(y, label_dict, a)
            # Set title
            if t is not None:
                a.set_title(t)
    else:
        raise NotImplementedError(
            "This function does not support the given parameters."
        )

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
