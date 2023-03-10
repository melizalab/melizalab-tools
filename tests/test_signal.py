# -*- coding: utf-8 -*-
# -*- mode: python -*-
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
    return signal.kernel(request.param, bandwidth, binsize)


def test_kernel_scale(conv_kernel):
    k, kt = conv_kernel
    assert k.size == kt.size
    assert k.sum() * binsize == pytest.approx(1.0)
