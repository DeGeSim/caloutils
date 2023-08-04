from typing import Optional

import numpy as np
import torch


def scale_b_to_a(a, b):
    assert not a.requires_grad
    mean, std = a.mean(), a.std()
    assert (std > 1e-6).all()
    sa = (a - mean) / (std + 1e-4)
    sb = (b - mean) / (std + 1e-4)
    return sa, sb


def sw1(s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    assert s.dim() == g.dim() == 1
    sa, sb = scale_b_to_a(s, g)
    return wasserstein_distance(sa, sb)


def cdf(arr, weigths):
    val, sortidx = arr.sort()
    cdf = val.cumsum(-1)
    cdf /= cdf[-1].clone()
    if weigths is None:
        return cdf
    else:
        return cdf, weigths[sortidx]


def cdf_by_hist(
    arr: torch.Tensor, bins: torch.Tensor, weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    dev = arr.device
    if weight is not None:
        weight = weight.cpu()
    val = torch.histogram(arr.cpu(), bins=bins.cpu(), weight=weight)
    cdf = val.hist.cumsum(-1)
    cdf /= cdf[-1].clone()
    return cdf.to(dev)


def energy_distance(u_values, v_values, u_weights=None, v_weights=None):
    return _cdf_distance(2, u_values, v_values, u_weights, v_weights)


def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
    return _cdf_distance(1, u_values, v_values, u_weights, v_weights)


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


def calc_sW1_dist(r: torch.Tensor, f: torch.Tensor, **kwargs) -> np.ndarray:
    dists = []
    assert r.shape == f.shape
    for iftx in range(r.shape[-1]):
        cdfdist = sw1(r[..., iftx], f[..., iftx])
        dists.append(cdfdist)
    return np.stack(dists, axis=0)


def calc_cdf_dist(r: torch.Tensor, f: torch.Tensor, **kwargs) -> np.ndarray:
    assert r.shape == f.shape
    dists = []
    for iftx in range(r.shape[-1]):
        real_cdf = cdf(r[..., iftx])
        fake_cdf = cdf(f[..., iftx])

        cdfdist = (fake_cdf - real_cdf).abs().mean(0)
        dists.append(cdfdist)
    return torch.stack(dists, dim=0).cpu().numpy()


def calc_wcdf_dist(
    r: torch.Tensor, f: torch.Tensor, rw: torch.Tensor, fw: torch.Tensor, **kwargs
) -> np.ndarray:
    assert r.shape == f.shape
    dists = []
    for iftx in range(r.shape[-1]):
        cdf_real, w_real = cdf(r[..., iftx], rw)
        cdf_fake, w_fake = cdf(f[..., iftx], fw)

        ww = w_fake * w_real
        ww /= ww.sum()

        cdfdist = ((cdf_fake - cdf_real) * ww).abs().mean(0)
        dists.append(cdfdist)
    return torch.stack(dists, dim=0).cpu().numpy()


def calc_hist_dist(
    r: torch.Tensor,
    f: torch.Tensor,
    bins: torch.Tensor,
    rw: Optional[torch.Tensor] = None,
    fw: Optional[torch.Tensor] = None,
) -> np.ndarray:
    dists = []
    assert r.shape == f.shape
    if len(r.shape) == 1:
        r = r.unsqueeze(-1)
        f = f.unsqueeze(-1)
    for iftx in range(r.shape[-1]):
        cdf_r = cdf_by_hist(r[..., iftx], bins[iftx], rw)
        cdf_f = cdf_by_hist(f[..., iftx], bins[iftx], fw)

        dist = (cdf_f - cdf_r).abs().mean()

        dists.append(dist)
    return torch.stack(dists, dim=0).cpu().numpy()
