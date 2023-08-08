from typing import Optional

import numpy as np
import torch

from .wasserstein import wasserstein_distance


def scale_b_to_a(a, b):
    """
    Scale tensor b to the same mean and standard deviation as tensor a.

    Args:
        a (torch.Tensor): Target tensor for scaling.
        b (torch.Tensor): Source tensor to be scaled.

    Returns:
        torch.Tensor: Scaled version of tensor b.
    """
    assert not a.requires_grad
    mean, std = a.mean(), a.std()
    assert (std > 1e-6).all()
    sa = (a - mean) / (std + 1e-4)
    sb = (b - mean) / (std + 1e-4)
    return sa, sb


def scaled_w1_distance(
    r: torch.Tensor,
    f: torch.Tensor,
    rw: Optional[torch.Tensor] = None,
    fw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate the scaled Wasserstein-1 Distance between two 1D samples.

    The scaled Wasserstein-1 Distance measures the difference between the distributions of two samples
    while accounting for differences in scale and location.

    Args:
        r (torch.Tensor): Reference sample.
        f (torch.Tensor): Comparison sample.
        rw (Optional[torch.Tensor]): Weights for the reference sample (optional).
        fw (Optional[torch.Tensor]): Weights for the comparison sample (optional).

    Returns:
        torch.Tensor: Calculated scaled Wasserstein-1 Distance between the two samples.
    """
    assert r.dim() == f.dim() == 1
    rs, fs = scale_b_to_a(r, f)
    return wasserstein_distance(rs, fs, rw, fw)


def calc_sw1_dist(
    r: torch.Tensor,
    f: torch.Tensor,
    rw: Optional[torch.Tensor] = None,
    fw: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Calculate scaled Wasserstein-1 distance metrics between two samples over multiple dimensions.

    Args:
        r (torch.Tensor): Reference sample.
        f (torch.Tensor): Comparison sample.
        rw (Optional[torch.Tensor]): Weights for the reference sample (optional).
        fw (Optional[torch.Tensor]): Weights for the comparison sample (optional).

    Returns:
        np.ndarray: Array of scaled Wasserstein-1 distance metrics calculated for each dimension.
    """
    assert r.shape == f.shape
    assert (rw is None) == (fw is None)
    dists = []
    for iftx in range(r.shape[-1]):
        cdfdist = scaled_w1_distance(r[..., iftx], f[..., iftx], rw, fw)
        dists.append(cdfdist)
    return np.stack(dists, axis=0)
