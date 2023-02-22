# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Common functions for plotting things """
import numpy as np


def spectrogram(
    ax,
    wav_file,
    window_s=0.020,
    shift_s=0.010,
    frequency_range=(700, 10000),
    compression=0.1,
    **plot_kwargs
):
    """Plot a spectrogram of a wave file, with some useful defaults for birdsong"""
    from math import ceil, log
    from ewave import wavfile, rescale
    from libtfr import mfft_precalc, fgrid, tgrid

    with wavfile(str(wav_file)) as fp:
        data = rescale(fp.read(), np.float64)
        sampling_rate_hz = fp.sampling_rate
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
