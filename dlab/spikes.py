# -*- mode: python -*-
"""Functions for processing spike trains (point processes)"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py as h5
import numpy as np
from numpy.typing import ArrayLike

# the mean waveform is computed from spikes upsampled by this factor and
# realigned to their trough (or peak) within this many samples (at the original
# rate) of peak_index, in chunks of this many spikes (see mean_waveform)
MEAN_UPSAMPLE = 3
MEAN_JITTER = 3
MEAN_CHUNK = 2000


def psth(
    spikes: ArrayLike,
    binwidth: float,
    *,
    start: float | None = None,
    stop: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a PSTH of spike times.

    spikes: array or list of spike times
    binwidth: the bin duration (same units as spikes)
    start: the start of the observation interval. If None, the time of the first spike is used.
    stop: the end of the observation interval. If None, the time of the last spike plus one bin is used.

    The interval is divided into as many whole bins as fit between start and
    stop (a partial bin at the end is dropped). Bins are half-open, [t, t +
    binwidth), including the last, so a spike at `stop` is not counted. Spike
    times that fall on a bin edge to within floating-point error (common when
    times are multiples of the sampling interval) are counted in the bin that
    starts there.

    Returns (spike counts, bin start times)
    """
    spikes = np.asarray(spikes, dtype=float).ravel()
    t1 = start if start is not None else spikes.min()
    t2 = stop if stop is not None else (spikes.max() + binwidth)
    # tolerance for floating-point error in bin positions, as a fraction of a bin
    eps = 1e-9
    nbins = max(0, int(np.floor((t2 - t1) / binwidth + eps)))
    idx = np.floor((spikes - t1) / binwidth + eps).astype(int)
    idx = idx[(idx >= 0) & (idx < nbins)]
    counts = np.bincount(idx, minlength=nbins)
    return counts, t1 + binwidth * np.arange(nbins)


def rate(
    spikes: ArrayLike,
    binwidth: float,
    kernel: ArrayLike,
    *,
    start: float | None = None,
    stop: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a smoothed estimate of spike rate from spike times.

    spikes: array or list of spike times
    binwidth: the bin duration (same units as spikes)
    kernel: the smoothing kernel. Sampling interval needs to be same as binwidth.
    start: the start of the observation interval. If None, the time of the first spike is used.
    stop: the end of the observation interval. If None, the time of the last spike plus one bin is used.

    Returns (rate estimate, bin times)
    """
    counts, bins = psth(spikes, binwidth, start=start, stop=stop)
    return np.convolve(counts, np.asarray(kernel, dtype=float), mode="same"), bins


@dataclass
class SpikeWaveforms:
    """
    waveforms: (nspikes, npoints) array
    times: nspikes array (times of spikes in units of samples)
    sampling_rate: the sampling rate of the spikes and the spike times (in Hz)
    peak_index: the index corresponding to the time of the spike in the waveform
    """

    waveforms: ArrayLike
    times: ArrayLike
    sampling_rate: int
    peak_index: int


@dataclass
class MeanWaveform:
    """
    mean: npoints array, the mean of a unit's spike waveforms, upsampled and
          realigned (see mean_waveform)
    sd: npoints array, their standard deviation
    n_spikes: the number of spikes averaged
    sampling_rate: the sampling rate of the mean (in Hz), after upsampling
    peak_index: the index of the alignment point (the trough or peak)
    upsample: the factor by which the spikes were upsampled
    aligned_to: 'trough' or 'peak'
    """

    mean: np.ndarray
    sd: np.ndarray
    n_spikes: int
    sampling_rate: float
    peak_index: int
    upsample: int = 1
    aligned_to: str = "trough"


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
    typical recording parameters. The mean and standard deviation of the
    waveforms, upsampled and realigned, are stored too (`mean_waveform` and
    `sd_waveform`, with their sampling rate, alignment point, upsampling factor
    and number of spikes; schema version 2), so that analyses of spike shape
    needn't load every spike (see mean_waveform and load_mean_waveform).

    """
    spikes = np.asarray(waveforms.waveforms)
    times = np.asarray(waveforms.times)
    nspikes, _npoints = spikes.shape
    if nspikes != times.size:
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
        if nspikes > 0:
            mean = mean_waveform(spikes, waveforms.sampling_rate, waveforms.peak_index)
            for name, data in (("mean_waveform", mean.mean), ("sd_waveform", mean.sd)):
                dset = fp.create_dataset(name, data=data)
                dset.attrs["sampling_rate"] = mean.sampling_rate
                dset.attrs["peak_index"] = mean.peak_index
                dset.attrs["n_spikes"] = mean.n_spikes
                dset.attrs["upsample"] = mean.upsample
                dset.attrs["aligned_to"] = mean.aligned_to

        fp.attrs["schema"] = "meliza-org.spikewaveforms"
        fp.attrs["schema_ver"] = 2
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


def mean_waveform(
    waveforms: ArrayLike,
    sampling_rate: float,
    peak_index: int,
    *,
    upsample: int = MEAN_UPSAMPLE,
    jitter: int = MEAN_JITTER,
) -> MeanWaveform:
    """The mean and standard deviation of a unit's spike waveforms, (nspikes,
    npoints), after upsampling each spike by `upsample` (bandwidth-limited, by
    FFT) and realigning it to its trough, or its peak if the spikes are
    positive at peak_index, within `jitter` samples of peak_index
    (quickspikes.tools.realign_spikes). Realigning removes the sub-sample
    jitter of the spike times, which would otherwise blur and shrink the mean.
    The result is shorter by `jitter` samples (at the original rate) at each
    end; its peak_index is the alignment point. Spikes are processed in chunks
    (MEAN_CHUNK), so waveforms may be an hdf5 dataset of any size."""
    import quickspikes.tools as qst

    nspikes = waveforms.shape[0]
    positive = float(np.mean(np.asarray(waveforms[:, peak_index], dtype="d"))) > 0
    align_by = (lambda x: x.argmax(-1)) if positive else (lambda x: x.argmin(-1))
    total = squares = None
    for a in range(0, nspikes, MEAN_CHUNK):
        chunk = np.asarray(waveforms[a : a + MEAN_CHUNK], dtype="d")
        _, aligned = qst.realign_spikes(
            np.zeros(chunk.shape[0]),
            chunk,
            upsample=upsample,
            jitter=jitter,
            expected_peak=peak_index,
            align_by=align_by,
        )
        if total is None:
            total = np.zeros(aligned.shape[1])
            squares = np.zeros(aligned.shape[1])
        total += aligned.sum(0)
        squares += (aligned**2).sum(0)
    mean = total / nspikes
    sd = np.sqrt(np.maximum(squares / nspikes - mean**2, 0.0))
    return MeanWaveform(
        mean,
        sd,
        nspikes,
        sampling_rate * upsample,
        (peak_index - jitter) * upsample,
        upsample,
        "peak" if positive else "trough",
    )


def load_mean_waveform(path: Path) -> MeanWaveform:
    """Load the mean spike waveform of a unit from its waveform file: the
    stored mean (schema version 2 and later), or for older files, computed from
    the stored waveforms in the same way (see mean_waveform)."""
    with h5.File(path, "r") as fp:
        if "mean_waveform" in fp:
            dset = fp["mean_waveform"]
            return MeanWaveform(
                dset[:],
                fp["sd_waveform"][:],
                int(dset.attrs["n_spikes"]),
                dset.attrs["sampling_rate"],
                int(dset.attrs["peak_index"]),
                int(dset.attrs["upsample"]),
                str(dset.attrs["aligned_to"]),
            )
        dset = fp["waveforms"]
        return mean_waveform(
            dset, dset.attrs["sampling_rate"], int(dset.attrs["peak_index"])
        )
