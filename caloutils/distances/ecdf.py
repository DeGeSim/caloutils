"""
ecdf: Calculate an approximation to the cumulative distribution function (CDF).
ecdf_distance: Calculate a distance metric between two 1D samples using ecdf.
calc_ecdf_dist: Convenience function to run ecdf_distance over multiple dimensions and convert the result to a numpy array.
"""

from typing import Optional

import numpy as np
import torch


def ecdf(
    arr: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Calculate an approximation to the cumulative distribution function (CDF) using empirical distribution.


    Args:
        arr (torch.Tensor): Input tensor containing data points.
        weights (Optional[torch.Tensor]): Weights for each data point (optional).

    Returns:
        tuple[torch.Tensor, Optional[torch.Tensor]]: Tuple containing the calculated CDF values and weights (if provided).
    """
    val, sortidx = arr.sort()
    cdf = val.cumsum(-1)
    cdf = cdf.clone() / cdf[-1].clone()
    if weights is None:
        return cdf, None
    else:
        return cdf, weights[sortidx]


def ecdf_distance(
    r: torch.Tensor,
    f: torch.Tensor,
    rw: Optional[torch.Tensor] = None,
    fw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate a distance metric between two 1D samples using the ecdf function.

    Args:
        r (torch.Tensor): Reference sample.
        f (torch.Tensor): Comparison sample.
        rw (Optional[torch.Tensor]): Weights for the reference sample (optional).
        fw (Optional[torch.Tensor]): Weights for the comparison sample (optional).

    Returns:
        torch.Tensor: Calculated distance metric between the two samples.
    """
    real_cdf, w_real = ecdf(r, rw)
    cdf_fake, w_fake = ecdf(f, fw)
    dist = cdf_fake - real_cdf
    if rw is not None:
        ww = w_fake * w_real
        ww /= ww.sum()
        dist *= ww
    return dist.abs().mean(0)


def calc_ecdf_dist(
    r: torch.Tensor,
    f: torch.Tensor,
    rw: Optional[torch.Tensor] = None,
    fw: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Calculate distance metrics between two samples over multiple dimensions using ecdf_distance and convert to a numpy array.

    Args:
        r (torch.Tensor): Reference sample.
        f (torch.Tensor): Comparison sample.
        rw (Optional[torch.Tensor]): Weights for the reference sample (optional).
        fw (Optional[torch.Tensor]): Weights for the comparison sample (optional).

    Returns:
        np.ndarray: Array of distance metrics calculated for each dimension.
    """
    assert r.shape == f.shape
    assert (rw is None) == (fw is None)
    dists = []
    for iftx in range(r.shape[-1]):
        dists.append(ecdf_distance(r[..., iftx], f[..., iftx], rw, fw))
    return torch.stack(dists, dim=0).cpu().numpy()
