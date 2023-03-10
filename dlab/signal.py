# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Signal processing functions """
from typing import Tuple

import numpy as np


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
