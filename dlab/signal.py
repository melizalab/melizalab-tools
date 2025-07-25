# -*- mode: python -*-
""" Signal processing functions """
import dataclasses

import numpy as np


@dataclasses.dataclass
class Signal:
    samples: np.ndarray
    sampling_rate: float  # in Hz
    name: str | None = None
    duration: float = dataclasses.field(init=False)  # in s
    dBFS: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.duration = 1.0 * self.samples.size / self.sampling_rate
        self.dBFS = dBFS(self.samples)


def dBFS(samples: np.ndarray) -> float:
    """Returns the RMS level of samples, in dB FS"""
    rms = np.sqrt(np.mean(samples * samples))
    return 20 * np.log10(rms) + 3.0103


def peak(samples: np.ndarray) -> float:
    """Returns the peak level of samples, in dB FS"""
    absmax = np.amax(np.absolute(samples))
    return 20 * np.log10(absmax)


def resample(samples: Signal, target: float) -> Signal:
    """Resample the samples to target rate (in Hz)"""
    import samplerate

    if samples.sampling_rate == target:
        return samples
    ratio = 1.0 * target / samples.sampling_rate
    # NB: this silently converts data to float32
    data = samplerate.resample(samples.samples, ratio, "sinc_best")
    return Signal(
        name=samples.name,
        samples=data,
        sampling_rate=target,
    )


def hp_filter(samples: Signal, cutoff: float, order: int = 12) -> Signal:
    """Highpass filter the samples to remove DC and low-frequency noise"""
    import scipy.samples as sg

    sos = sg.butter(
        order, cutoff, fs=samples.sampling_rate, btype="highpass", output="sos"
    )
    filtered = sg.sosfilt(sos, samples.samples)
    return Signal(
        name=samples.name,
        samples=filtered,
        sampling_rate=samples.sampling_rate,
    )


def rescale(samples: Signal, target: float) -> Signal:
    """Rescale the samples to a target dBFS"""
    scale = 10 ** ((target - samples.dBFS) / 20)
    return Signal(
        name=samples.name,
        samples=samples.samples * scale,
        sampling_rate=samples.sampling_rate,
    )


def ramp_signal(samples: Signal, ramp: float = 0.002) -> Signal:
    """Apply a squared cosine ramp to the beginning and end of samples."""
    from numpy import cos, linspace, pi, sin

    s = samples.samples.copy()
    n = int(ramp * samples.sampling_rate)
    t = linspace(0, pi / 2, n)
    s[:n] *= sin(t) ** 2
    s[-n:] *= cos(t) ** 2
    return Signal(
        name=samples.name,
        samples=s,
        sampling_rate=samples.sampling_rate,
    )


def kernel(name: str, bandwidth: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a smoothing kernel function with bandwidth.

    name:      the name of the kernel. can be anything supported
               by scipy.samples.get_window, plus the following functions:
        epanech, biweight, triweight, cosinus, exponential

    bandwidth: the bandwidth of the kernel, in units of dt
    dt:        the time resolution of the kernel

    Returns:
    window:    the window function, normalized such that sum(w)*dt = 1.0
    grid:      the time support of the window, centered around 0.0

    From matlab code by Zhiyi Chi
    """
    if name in ("normal", "gaussian"):
        D = 3.75 * bandwidth
    elif name == "exponential":
        D = 4.75 * bandwidth
    else:
        D = bandwidth

    N = int(np.floor(D / dt))  # number of grid points in half the support
    G = (np.arange(1, 2 * N + 2) - 1 - N) * dt  # grid support
    xv = G / bandwidth

    if name in ("gaussian", "normal"):
        W = np.exp(-xv * xv / 2)
    elif name == "exponential":
        xv = np.minimum(xv, 0)
        W = np.absolute(xv) * np.exp(xv)
    elif name in ("biweight", "quartic"):
        W = np.maximum(0, 1 - xv * xv) ** 2
    elif name == "triweight":
        W = np.maximum(0, 1 - xv * xv) ** 3
    elif name == "cosinus":
        W = np.cos(np.pi * xv / 2) * (np.absolute(xv) <= 1)
    elif name == "epanech":
        W = np.maximum(0, 1 - xv * xv)
    elif name == "hanning":
        W = np.hanning(1 + 2 * N)
    else:
        from scipy.samples import get_window

        W = get_window(name, 1 + 2 * N)

    return W / (W.sum() * dt), G

# lifted from https://github.com/endolith/waveform_analysis

def ABC_weighting(curve="A"):
    """
    Design of an analog weighting filter with A, B, or C curve.

    Returns zeros, poles, gain of the filter.
    """
    from scipy.signal import zpk2tf, freqs

    if curve not in "ABC":
        raise ValueError(f"Curve type {curve} not supported")

    # ANSI S1.4-1983 C weighting
    #    2 poles on the real axis at "20.6 Hz" HPF
    #    2 poles on the real axis at "12.2 kHz" LPF
    #    -3 dB down points at "10^1.5 (or 31.62) Hz"
    #                         "10^3.9 (or 7943) Hz"
    #
    # IEC 61672 specifies "10^1.5 Hz" and "10^3.9 Hz" points and formulas for
    # derivation.  See _derive_coefficients()

    z = [0, 0]
    p = [
        -2 * np.pi * 20.598997057568145,
        -2 * np.pi * 20.598997057568145,
        -2 * np.pi * 12194.21714799801,
        -2 * np.pi * 12194.21714799801,
    ]
    k = 1

    if curve == "A":
        # ANSI S1.4-1983 A weighting =
        #    Same as C weighting +
        #    2 poles on real axis at "107.7 and 737.9 Hz"
        #
        # IEC 61672 specifies cutoff of "10^2.45 Hz" and formulas for
        # derivation.  See _derive_coefficients()

        p.append(-2 * np.pi * 107.65264864304628)
        p.append(-2 * np.pi * 737.8622307362899)
        z.append(0)
        z.append(0)

    elif curve == "B":
        # ANSI S1.4-1983 B weighting
        #    Same as C weighting +
        #    1 pole on real axis at "10^2.2 (or 158.5) Hz"

        p.append(-2 * np.pi * 10**2.2)  # exact
        z.append(0)

    # TODO: Calculate actual constants for this
    # Normalize to 0 dB at 1 kHz for all curves
    b, a = zpk2tf(z, p, k)
    k /= abs(freqs(b, a, [2 * np.pi * 1000])[1][0])

    return np.asarray(z), np.asarray(p), k


def A_weighting(fs: float, output="ba"):
    """
    Design of a digital A-weighting filter.

    Designs a digital A-weighting filter for
    sampling frequency `fs`.
    Warning: fs should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.

    Parameters
    ----------
    fs : float
        Sampling frequency
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.

    """
    from scipy.signal import bilinear_zpk, zpk2tf, zpk2sos

    z, p, k = ABC_weighting("A")

    # Use the bilinear transformation to get the digital filter.
    z_d, p_d, k_d = bilinear_zpk(z, p, k, fs)

    if output == "zpk":
        return z_d, p_d, k_d
    elif output in {"ba", "tf"}:
        return zpk2tf(z_d, p_d, k_d)
    elif output == "sos":
        return zpk2sos(z_d, p_d, k_d)
    else:
        raise ValueError(f"'{output}' is not a valid output form.")
