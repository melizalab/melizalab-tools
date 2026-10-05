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


def test_ramp_signal_shapes_ends():
    """The ramp takes the first and last samples to zero and leaves the middle
    of the signal alone."""
    sig = signal.Signal(np.ones(1000), 1000)
    ramped = signal.ramp_signal(sig, duration_s=0.01)
    assert ramped.samples[0] == pytest.approx(0) and ramped.samples[
        -1
    ] == pytest.approx(0)
    assert (ramped.samples[10:-10] == 1).all(), "middle unchanged"
    assert (sig.samples == 1).all(), "input not modified"


def test_ramp_signal_shorter_than_one_sample():
    """A ramp that rounds to zero samples leaves the signal unchanged (it used
    to raise, because s[-0:] is the whole array)."""
    sig = signal.Signal(np.ones(100), 1000)
    assert (signal.ramp_signal(sig, duration_s=0.0001).samples == 1).all()


@pytest.mark.parametrize("curve", ["", "AB", "D"])
def test_abc_weighting_rejects_unknown_curves(curve):
    """Only "A", "B" and "C" are accepted (substrings of "ABC" used to be)."""
    with pytest.raises(ValueError):
        signal.ABC_weighting(curve)
