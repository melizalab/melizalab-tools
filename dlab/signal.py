# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Signal processing functions """
import dataclasses
from typing import Tuple, Optional

import numpy as np


@dataclasses.dataclass
class Signal:
    signal: np.ndarray
    sampling_rate: float  # in Hz
    name: Optional[str] = None
    duration: float = dataclasses.field(init=False)  # in s
    dBFS: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.duration = 1.0 * self.signal.size / self.sampling_rate
        self.dBFS = dBFS(self.signal)


def kernel(name: str, bandwidth: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a smoothing kernel function with bandwidth.

    name:      the name of the kernel. can be anything supported
               by scipy.signal.get_window, plus the following functions:
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
        from scipy.signal import get_window

        W = get_window(name, 1 + 2 * N)

    return W / (W.sum() * dt), G


def dBFS(signal: np.ndarray) -> float:
    """Returns the RMS level of signal, in dB FS"""
    rms = np.sqrt(np.mean(signal * signal))
    return 20 * np.log10(rms) + 3.0103


def peak(signal: np.ndarray) -> float:
    """Returns the peak level of signal, in dB FS"""
    absmax = np.amax(np.absolute(signal))
    return 20 * np.log10(absmax)


def resample(signal: Signal, target: float) -> Signal:
    """Resample the signal to target rate (in Hz)"""
    import samplerate

    if signal.sampling_rate == target:
        return signal
    ratio = 1.0 * target / signal.sampling_rate
    # NB: this silently converts data to float32
    data = samplerate.resample(signal.signal, ratio, "sinc_best")
    return Signal(
        name=signal.name,
        signal=data,
        sampling_rate=target,
    )


def hp_filter(signal: Signal, cutoff: float, order: int = 12) -> Signal:
    """Highpass filter the signal to remove DC and low-frequency noise"""
    import scipy.signal as sg

    sos = sg.butter(
        order, cutoff, fs=signal.sampling_rate, btype="highpass", output="sos"
    )
    filtered = sg.sosfilt(sos, signal.signal)
    return Signal(
        name=signal.name,
        signal=filtered,
        sampling_rate=signal.sampling_rate,
    )


def rescale(signal: Signal, target: float) -> Signal:
    """Rescale the signal to a target dBFS"""
    scale = 10 ** ((target - signal.dBFS) / 20)
    return Signal(
        name=signal.name,
        signal=signal.signal * scale,
        sampling_rate=signal.sampling_rate,
    )


def ramp_signal(signal: Signal, ramp: float = 0.002) -> Signal:
    """Apply a squared cosine ramp to the beginning and end of signal."""
    from numpy import linspace, pi, sin, cos

    s = signal.signal.copy()
    n = int(ramp * signal.sampling_rate)
    t = linspace(0, pi / 2, n)
    s[:n] *= sin(t) ** 2
    s[-n:] *= cos(t) ** 2
    return Signal(
        name=signal.name,
        signal=s,
        sampling_rate=signal.sampling_rate,
    )
