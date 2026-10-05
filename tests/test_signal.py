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


def test_dbfs_of_full_scale_sine_is_zero():
    """dBFS is RMS relative to a full-scale sine (hence the +3.01 dB)."""
    t = np.arange(48000) / 48000
    assert signal.dBFS(np.sin(2 * np.pi * 100 * t)) == pytest.approx(0, abs=1e-6)


def test_peak():
    """peak is the largest absolute sample, in dB relative to 1.0."""
    assert signal.peak(np.array([0.1, -0.5, 0.2])) == pytest.approx(-6.0206, abs=1e-4)


def test_write_wav_round_trip(tmp_path):
    """Float samples are written as int16 and read back to within one count."""
    import ewave

    samples = 0.5 * np.sin(np.arange(4800) / 10)
    signal.Signal(samples, 48000).write_wav(tmp_path / "x.wav")
    with ewave.open(tmp_path / "x.wav") as fp:
        assert fp.sampling_rate == 48000 and fp.dtype == np.dtype("int16")
        data = ewave.rescale(fp.read(), "d")
    assert data == pytest.approx(samples, abs=1 / 32767)


def test_resample_to_same_rate_returns_input(test_signal):
    """Resampling to the current rate is a no-op that returns the same object."""
    assert signal.resample(test_signal, target=48000) is test_signal


def test_resample_converts_to_float32(test_signal):
    """NB: samplerate returns float32, whatever the input dtype."""
    assert signal.resample(test_signal, target=24000).samples.dtype == np.float32


def test_kernel_support(conv_kernel):
    """Kernels have an odd number of points on a grid centred on zero."""
    k, kt = conv_kernel
    assert k.size % 2 == 1
    assert kt[k.size // 2] == pytest.approx(0)
    assert kt == pytest.approx(-kt[::-1]), "grid symmetric about zero"


@pytest.mark.parametrize("name", [n for n in kernels if n != "exponential"])
def test_symmetric_kernels(name):
    """All kernels except exponential are symmetric and peak at zero."""
    k, kt = signal.smoothing_kernel(name, bandwidth, binsize)
    assert k == pytest.approx(k[::-1])
    assert kt[k.argmax()] == pytest.approx(0)


def test_gaussian_kernel_extent():
    """The gaussian kernel extends to 3.75 bandwidths on each side."""
    _, kt = signal.smoothing_kernel("gaussian", bandwidth, binsize)
    assert kt[-1] == pytest.approx(3.75 * bandwidth, abs=binsize)


def test_exponential_kernel_is_one_sided():
    """The exponential kernel is nonzero only for t < 0 and peaks at
    -bandwidth."""
    k, kt = signal.smoothing_kernel("exponential", bandwidth, binsize)
    assert (k[kt >= 0] == 0).all()
    assert kt[k.argmax()] == pytest.approx(-bandwidth)


def test_kernel_falls_back_to_scipy_windows():
    """Names not built in are passed to scipy.signal.get_window."""
    k, _ = signal.smoothing_kernel("boxcar", bandwidth, binsize)
    assert np.allclose(k, k[0]), "a boxcar is flat"
    assert k.sum() * binsize == pytest.approx(1.0)


def test_kernel_unknown_name():
    with pytest.raises(ValueError):
        signal.smoothing_kernel("nonsense", bandwidth, binsize)


def response_db(ba_or_sos, freq, fs, sos=False):
    from scipy.signal import freqz, sosfreqz

    if sos:
        _, h = sosfreqz(ba_or_sos, worN=[freq], fs=fs)
    else:
        _, h = freqz(*ba_or_sos, worN=[freq], fs=fs)
    return 20 * np.log10(abs(h[0]))


@pytest.mark.parametrize(
    "freq,nominal,lower,upper",
    [(100, -19.1, 1.0, 1.0), (1000, 0.0, 0.7, 0.7), (10000, -2.5, 3.0, 2.0)],
)
def test_a_weighting_class_1(freq, nominal, lower, upper):
    """At fs=48 kHz the digital A-weighting filter is within the IEC 61672-1
    class 1 tolerances of the nominal response. (At 10 kHz it is -3.7 dB;
    the bilinear transform pulls it down near Nyquist.)"""
    db = response_db(signal.A_weighting(48000), freq, 48000)
    assert nominal - lower <= db <= nominal + upper


def test_a_weighting_output_forms_agree():
    """'ba', 'sos' and 'zpk' describe the same filter."""
    from scipy.signal import zpk2sos

    fs = 48000
    ba = response_db(signal.A_weighting(fs, "ba"), 1000, fs)
    sos = response_db(signal.A_weighting(fs, "sos"), 1000, fs, sos=True)
    zpk = response_db(zpk2sos(*signal.A_weighting(fs, "zpk")), 1000, fs, sos=True)
    assert ba == pytest.approx(sos, abs=0.01) and sos == pytest.approx(zpk, abs=0.01)


def test_a_weighting_invalid_output():
    with pytest.raises(ValueError):
        signal.A_weighting(48000, "nonsense")
