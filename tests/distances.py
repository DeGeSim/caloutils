from functools import partial

import torch

from caloutils.distances import (
    calc_ecdf_dist,
    calc_hist_dist,
    calc_sw1_dist,
    ecdf_distance,
    energy_distance,
    hist_distance,
    scaled_w1_distance,
    wasserstein_distance,
)


def test_distances():
    x = torch.tensor([1, 0, 0, 0]).float()
    y = torch.tensor([0, 1, 2, 0]).float()
    w = torch.tensor([1, 2, 1, 1]).float()
    bins = torch.arange(0, 4, step=0.5)

    for dist in [
        wasserstein_distance,
        energy_distance,
        ecdf_distance,
        partial(hist_distance, bins=bins),
        scaled_w1_distance,
    ]:
        assert dist(x, x, rw=w, fw=w) == 0
        assert dist(x, y, rw=w, fw=w) > 0

    x = x.unsqueeze(1)
    y = y.unsqueeze(1)

    for dist in [
        calc_ecdf_dist,
        calc_sw1_dist,
        partial(calc_hist_dist, bins=bins.unsqueeze(1)),
    ]:
        assert all((dist(x, x, rw=w, fw=w) == 0))
        assert all(dist(x, y, rw=w, fw=w) >= 0)
