from functools import partial

import numpy as np
import scipy
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .idxscale import IdxToScale

__safety_gap = 1e-6


def dequant(x):
    assert x.dtype in [np.float64, np.int32]
    noise = np.random.rand(*x.shape)
    xnew = x.astype("float64") + np.clip(noise, __safety_gap, 1 - __safety_gap)
    assert np.all(x == np.floor(xnew))
    return xnew


def requant(x):
    assert x.dtype == np.float64
    # x_copy = x.copy()
    x = np.floor(x).astype("int32")
    # with the clip in dequant, the input must change
    # x[x == x_copy] -= 1
    # assert np.all(x.astype("int") == x)
    # delta = np.abs((x_copy - x))
    # assert (delta > 0).all() and (delta <= 1).all()
    return x


def forward(x, lower, dist):
    assert x.dtype == np.float64
    return (x - lower) / dist


def backward(x, lower, dist):
    assert x.dtype == np.float64
    return x * dist + lower


def idt(x):
    return x


def dequant_stdscale(inputrange=None) -> list:
    if inputrange is None:
        scaletf = IdxToScale((0, 1))
    else:
        lower, upper = inputrange
        dist = upper - lower

        scaletf = FunctionTransformer(
            partial(forward, lower=lower, dist=dist),
            partial(backward, lower=lower, dist=dist),
            check_inverse=True,
            validate=True,
        )
    tfseq = [
        FunctionTransformer(dequant, requant, check_inverse=True, validate=True),
        scaletf,
        FunctionTransformer(
            scipy.special.logit,  # scipy.special.erf,  # ,
            scipy.special.expit,  # scipy.special.erfinv,  #
            check_inverse=True,
            validate=True,
        ),
        StandardScaler(),
    ]
    return tfseq
