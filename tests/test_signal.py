# -*- mode: python -*-
import numpy as np
import pytest

from dlab import signal

binsize = 0.01
bandwidth = 0.1
kernels = [
    "gaussian",
    "exponential",
    "biweight",
    "triweight",
    "cosinus",
    "epanech",
    "hanning",
]


@pytest.fixture(params=kernels)
def conv_kernel(request):
    return signal.smoothing_kernel(request.param, bandwidth, binsize)


@pytest.fixture
def test_signal():
    np.random.seed(1028)
    samples = np.random.randn(48000)
    samples *= 0.1 / samples.std()
    return signal.Signal(samples, 48000)


def test_kernel_scale(conv_kernel):
    k, kt = conv_kernel
    assert k.size == kt.size
    assert k.sum() * binsize == pytest.approx(1.0)


def test_signal_attributes(test_signal):
    expected_dBFS = -20 + 3.0103
    assert test_signal.duration == 1.0
    assert np.abs(test_signal.dBFS - expected_dBFS) < 0.001


def test_signal_resample(test_signal):
    resampled = signal.resample(test_signal, target=24000)
    assert resampled.sampling_rate == 24000
    assert resampled.duration == test_signal.duration
    # resampling will change the scale
    # assert resampled.dBFS == test_signal.dBFS


def test_signal_rescale(test_signal):
    target_dBFS = -40
    rescaled = signal.rescale(test_signal, target=target_dBFS)
    assert rescaled.sampling_rate == test_signal.sampling_rate
    assert rescaled.duration == test_signal.duration
    assert np.abs(rescaled.dBFS - target_dBFS) < 0.001


def test_hp_filter(test_signal):
    # add a DC offset
    test_signal.samples += 1
    assert test_signal.samples.mean() > 0
    filtered = signal.hp_filter(test_signal, cutoff_Hz=100)
    assert filtered.sampling_rate == test_signal.sampling_rate
    assert filtered.duration == test_signal.duration
    assert np.abs(filtered.samples.mean()) < 0.001
