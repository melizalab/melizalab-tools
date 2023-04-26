# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Functions for processing spike trains (point processes) """
from typing import Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

import h5py as h5
import numpy as np
from numpy.typing import ArrayLike


def psth(
    spikes: ArrayLike,
    binwidth: float,
    *,
    start: Optional[float] = None,
    stop: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a PSTH of spike times.

    spikes: array or list of spike times
    binwidth: the bin duration (same units as spikes)
    start: the start of the observation interval. If None, the time of the first spike is used.
    stop: the end of the observation interval. If None, the time of the last spike plus one bin is used.

    Returns (spike counts, bin times)
    """
    spikes = np.asarray(spikes)
    t1 = start if start is not None else spikes.min()
    t2 = stop if stop is not None else (spikes.max() + binwidth)
    bins = np.arange(t1, t2, binwidth)
    counts, bins = np.histogram(spikes, bins)
    return counts, bins[:-1]


def rate(
    spikes: ArrayLike,
    binwidth: float,
    kernel: ArrayLike,
    *,
    start: Optional[float] = None,
    stop: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a smoothed estimate of spike rate from spike times.

    spikes: array or list of spike times
    binwidth: the bin duration (same units as spikes)
    kernel: the smoothing kernel. Sampling interval needs to be same as binwidth.
    start: the start of the observation interval. If None, the time of the first spike is used.
    stop: the end of the observation interval. If None, the time of the last spike is used.

    Returns (rate estimate, bin times)
    """
    counts, bins = psth(spikes, binwidth, start=start, stop=stop)
    return np.convolve(counts, kernel, mode="same"), bins


@dataclass
class SpikeWaveforms:
    """
    waveform: nspikes x npoints array
    times: nspikes array (times of spikes in units of samples)
    sampling_rate: the sampling rate of the spikes and the spike times (in Hz)
    peak_index: the index corresponding to the time of the spike in the waveform
    """

    waveforms: ArrayLike
    times: ArrayLike
    sampling_rate: int
    peak_index: int


def save_waveforms(
    path: Path,
    waveforms: SpikeWaveforms,
    **attributes: Any,
) -> None:
    """Save spike waveforms to an hdf5 file

    path: the location of the file (will overwrite)
    attributes: any additional metadata to store in the file

    Waveforms should be stored as they were recorded. The `peak_index` argument
    will typically refer to the maximum negative deflection of the spike given
    typical recording parameters.

    """
    spikes = np.asarray(waveforms.waveforms)
    times = np.asarray(waveforms.times)
    ntimes, nspikes = spikes.shape
    if ntimes != times.size:
        raise ValueError(
            "number of rows in waveform array must match number of elements in times array"
        )
    with h5.File(path, "w") as fp:
        # unchunked storage uses the most space but allows random access
        dset_spikes = fp.create_dataset("waveforms", data=spikes)
        dset_spikes.attrs["sampling_rate"] = waveforms.sampling_rate
        dset_spikes.attrs["peak_index"] = waveforms.peak_index
        dset_times = fp.create_dataset("times", data=times)
        dset_times.attrs["sampling_rate"] = waveforms.sampling_rate

        fp.attrs["schema"] = "meliza-org.spikewaveforms"
        fp.attrs["schema_ver"] = 1
        for k, v in attributes.items():
            fp.attrs[k] = v


def load_waveforms(path: Path) -> SpikeWaveforms:
    """Load spike waveforms from an hdf5 file.

    Data are returned in a SpikeWaveforms structure, plus the top-level
    attributes in the "attrs" property. The data are loaded lazily, so the file
    handle remains open.

    """
    fp = h5.File(path, "r")
    dset_spikes = fp["waveforms"]
    dset_times = fp["times"]
    out = SpikeWaveforms(
        dset_spikes,
        dset_times,
        dset_spikes.attrs["sampling_rate"],
        dset_spikes.attrs["peak_index"],
    )
    out.attrs = dict(fp.attrs)
    return out
