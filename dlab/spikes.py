# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Functions for processing spike trains (point processes) """
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


def psth(
    spikes: ArrayLike,
    binwidth: float,
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
    counts, bins = psth(spikes, binwidth, start, stop)
    return np.convolve(counts, kernel, mode="same"), bins
