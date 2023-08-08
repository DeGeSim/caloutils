"""
histcdf: Calculate an approximation to the cumulative distribution function using histograms.
hist_distance: Calculate a distance metric between two 1D samples using histcdf.
calc_hist_dist: Convenience function to run hist_distance over multiple dimensions and convert the result to a numpy array.
"""

from typing import Optional

import numpy as np
import torch


def histcdf(
    arr: torch.Tensor, bins: torch.Tensor, weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculate an approximation to the cumulative distribution function (CDF) using histograms.

    Args:
        arr (torch.Tensor): Input tensor containing data points.
        bins (torch.Tensor): Bins for histogram calculation.
        weight (Optional[torch.Tensor]): Weights for each data point (optional).

    Returns:
        torch.Tensor: Approximated cumulative distribution function values.
    """
    dev = arr.device
    if weight is not None:
        weight = weight.cpu()
    val = torch.histogram(arr.cpu(), bins=bins.cpu(), weight=weight)
    cdf = val.hist.cumsum(-1)
    cdf /= cdf[-1].clone()
    return cdf.to(dev)


def hist_distance(
    r: torch.Tensor,
    f: torch.Tensor,
    bins: torch.Tensor,
    rw: Optional[torch.Tensor] = None,
    fw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate a distance metric between two 1D samples using the `histcdf` function.

    Args:
        r (torch.Tensor): Reference sample.
        f (torch.Tensor): Comparison sample.
        bins (torch.Tensor): Bins for histogram calculation.
        rw (Optional[torch.Tensor]): Weights for the reference sample (optional).
        fw (Optional[torch.Tensor]): Weights for the comparison sample (optional).

    Returns:
        torch.Tensor: Calculated distance metric between the two samples.
    """
    cdf_r = histcdf(r, bins, rw)
    cdf_f = histcdf(f, bins, fw)
    dist = (cdf_f - cdf_r).abs().mean()
    return dist


def calc_hist_dist(
    r: torch.Tensor,
    f: torch.Tensor,
    bins: torch.Tensor,
    rw: Optional[torch.Tensor] = None,
    fw: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Calculate distance metrics between two samples over multiple dimensions using `hist_distance` and convert to a numpy array.

    Args:
        r (torch.Tensor): Reference sample.
        f (torch.Tensor): Comparison sample.
        bins (torch.Tensor): Bins for histogram calculation.
        rw (Optional[torch.Tensor]): Weights for the reference sample (optional).
        fw (Optional[torch.Tensor]): Weights for the comparison sample (optional).

    Returns:
        np.ndarray: Array of distance metrics calculated for each dimension.
    """
    assert r.shape == f.shape
    assert (rw is None) == (fw is None)
    if len(r.shape) == 1:
        r = r.unsqueeze(-1)
        f = f.unsqueeze(-1)

    dists = []
    for iftx in range(r.shape[-1]):
        if bins.dim() == 1:
            ibins = bins
        elif bins.shape[-1] == r.shape[-1]:
            ibins = bins[:, iftx]
        elif bins.shape[0] == r.shape[-1]:
            ibins = bins[iftx]
        else:
            raise Exception("Bins have the wrong shape/dimension.")
        idist = hist_distance(r[..., iftx], f[..., iftx], ibins, rw, fw)
        dists.append(idist)

    return torch.stack(dists, dim=0).cpu().numpy()
