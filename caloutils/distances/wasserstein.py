"""
1D Sample Distance Metrics
"""

from typing import Optional

import torch


def energy_distance(
    r: torch.Tensor,
    f: torch.Tensor,
    rw: Optional[torch.Tensor] = None,
    fw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate the Energy Distance between two 1D samples.

    The Energy Distance is a statistical measure that quantifies the difference between the empirical distributions
    of two sets of data points.

    Args:
        r (torch.Tensor): Reference sample.
        f (torch.Tensor): Comparison sample.
        rw (Optional[torch.Tensor]): Weights for the reference sample (optional).
        fw (Optional[torch.Tensor]): Weights for the comparison sample (optional).

    Returns:
        torch.Tensor: Calculated Energy Distance between the two samples.
    """
    return _cdf_distance(2, r, f, rw, fw)


def wasserstein_distance(
    r: torch.Tensor,
    f: torch.Tensor,
    rw: Optional[torch.Tensor] = None,
    fw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate the Wasserstein Distance (Earth Mover's Distance) between two 1D samples.

    The Wasserstein Distance measures the minimum "cost" of transforming one distribution into another.

    Args:
        r (torch.Tensor): Reference sample.
        f (torch.Tensor): Comparison sample.
        rw (Optional[torch.Tensor]): Weights for the reference sample (optional).
        fw (Optional[torch.Tensor]): Weights for the comparison sample (optional).

    Returns:
        torch.Tensor: Calculated Wasserstein Distance between the two samples.
    """
    return _cdf_distance(1, r, f, rw, fw)


def _cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
    """Torch implementation of scipy/scipy/stats /_stats_py.py / _cdf_distance"""
    dev = u_values.device
    u_sorter = torch.argsort(u_values)
    v_sorter = torch.argsort(v_values)

    all_values = torch.cat((u_values, v_values))
    all_values, _ = torch.sort(all_values)

    # Compute the differences between pairs of successive values of u and v.
    deltas = torch.diff(all_values)

    # Get the respective positions of the values of u and v
    #  among the values of both distributions.
    u_cdf_indices = torch.searchsorted(
        u_values[u_sorter], all_values[:-1], right=True
    )
    v_cdf_indices = torch.searchsorted(
        v_values[v_sorter], all_values[:-1], right=True
    )

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices.float() / u_values.numel()
    else:
        u_sorted_cumweights = torch.cat(
            (torch.tensor([0.0]).to(dev), u_weights[u_sorter].cumsum(0))
        )
        u_cdf = u_sorted_cumweights[u_cdf_indices].float() / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices.float() / v_values.numel()
    else:
        v_sorted_cumweights = torch.cat(
            (torch.tensor([0.0]).to(dev), v_weights[v_sorter].cumsum(0))
        )
        v_cdf = v_sorted_cumweights[v_cdf_indices].float() / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    if p == 1:
        return torch.sum(torch.abs(u_cdf - v_cdf) * deltas)
    if p == 2:
        return torch.sqrt(torch.sum(torch.square(u_cdf - v_cdf) * deltas))
    return torch.pow(
        torch.sum(torch.pow(torch.abs(u_cdf - v_cdf), p) * deltas), 1 / p
    )
