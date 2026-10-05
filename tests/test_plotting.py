# -*- mode: python -*-
"""Tests for dlab.plotting. matplotlib is not a dependency, so spectrogram is
checked with a stand-in for the axes, and the helpers that need real axes are
not tested."""

import numpy as np
import pytest

from dlab import plotting


class FakeAxes:
    """Records the arguments to imshow"""

    def imshow(self, image, **kwargs):
        self.image = image
        self.kwargs = kwargs


@pytest.fixture
def tone():
    """1 s of a 2 kHz tone at 40 kHz"""
    rate = 40000
    return np.sin(2 * np.pi * 2000 * np.arange(rate) / rate), rate


def test_spectrogram_image(tone):
    """The image covers the requested frequency range over the whole signal,
    one column per 10 ms frame, and its peak is at the tone's frequency."""
    data, rate = tone
    ax = FakeAxes()
    plotting.spectrogram(ax, data, rate)
    _, _, f0, f1 = ax.kwargs["extent"]
    assert 700 <= f0 and f1 <= 10000, "default frequency range"
    assert ax.image.shape[1] == pytest.approx(100, abs=2), "10 ms shift"
    nfreq = ax.image.shape[0]
    peak = ax.image.mean(1).argmax()
    assert f0 + peak * (f1 - f0) / (nfreq - 1) == pytest.approx(2000, abs=100)
    assert ax.kwargs["origin"] == "lower" and ax.kwargs["aspect"] == "auto"


def test_spectrogram_passes_plot_kwargs(tone):
    data, rate = tone
    ax = FakeAxes()
    plotting.spectrogram(ax, data, rate, frequency_range=(1000, 5000), cmap="jet")
    assert ax.kwargs["cmap"] == "jet"
    assert ax.kwargs["extent"][2] >= 1000 and ax.kwargs["extent"][3] <= 5000
