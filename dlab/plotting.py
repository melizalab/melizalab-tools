# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Common functions for plotting things. 

Functions in this module should take matplotlib axes as their first argument,
but there is no direct import of matplotlib.

"""
from typing import Tuple

import numpy as np


def spectrogram(
    ax,
    signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    window_s: float = 0.020,
    shift_s: float = 0.010,
    frequency_range: Tuple[float, float] = (700, 10000),
    compression: float = 0.1,
    **plot_kwargs
):
    """Plot a spectrogram of a signal, with some useful defaults for birdsong"""
    from math import ceil, log

    from ewave import rescale
    from libtfr import fgrid, mfft_precalc, tgrid

    data = rescale(signal, np.float64)
    window_samples = int(window_s * sampling_rate_hz)
    nfft = 2 ** ceil(log(window_samples, 2))
    shift_samples = int(shift_s * sampling_rate_hz)
    mfft = mfft_precalc(nfft, np.hanning(window_samples))
    spec = mfft.mtspec(data, shift_samples)
    f, fi = fgrid(sampling_rate_hz, nfft, frequency_range)
    t = tgrid(spec, sampling_rate_hz, shift_samples)
    ax.imshow(
        np.log(spec[fi, :] + compression) - np.log(compression),
        aspect="auto",
        origin="lower",
        extent=(t[0], t[-1], f[0], f[-1]),
        **plot_kwargs
    )


def simple_axes(*axes):
    """Simple axes: only bottom and right lines shown"""
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()


def hide_axes(*axes):
    """Hidden axes: no axis lines shown"""
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_frame_on(False)


def adjust_raster_ticks(ax, gap=0):
    """Adjust raster marks to have gap pixels between them (sort of)"""
    miny, maxy = ax.get_ylim()
    ht = ax.get_window_extent().height
    for p in ax.lines:
        p.set_markersize(ht / ((maxy - miny)) - gap)
