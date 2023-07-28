import torch
from fgsim.config import conf
from torch_geometric.data import Batch
from torch_geometric.nn.pool import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_scatter import scatter_std

n_layers = 45


def analyze_layers(batch: Batch) -> dict[str, torch.Tensor]:
    """Get the ratio between half-turnon and
    half turnoff to peak center from a PC"""
    batchidx = batch.batch
    n_events = int(batchidx[-1] + 1)
    device = batch.x.device
    z = batch.xyz[..., -1]

    # compute the histogramm with dim (events,layers)
    z_ev = (z.long() + batchidx * n_layers).sort().values

    zvals, zcounts = z_ev.unique_consecutive(return_counts=True)

    layer_idx = zvals % n_layers
    ev_idx = (zvals - layer_idx) // n_layers

    hist = torch.zeros(n_events, n_layers).long().to(device)
    hist[ev_idx, layer_idx] = zcounts

    assert (hist.sum(1) == batch.ptr[1:] - batch.ptr[:-1]).all()

    max_hits_per_event, peak_layer_per_event = hist.max(1)

    # produce an index for each layer in each event
    # centered_layers_perevent = (
    #     torch.arange(n_layers).to(device).reshape(1, -1).repeat(n_events, 1)
    # )
    # # shift peak to number of layers
    # centered_layers_perevent += -peak_layer_per_event.reshape(-1, 1) + n_layers
    # # separate idx by event
    # eventshift = (
    #     (torch.arange(n_events) * n_layers * 2)
    #     .to(device)
    #     .reshape(-1, 1)
    #     .repeat(1, n_layers)
    # )
    # centered_layers_perevent += eventshift

    # # aggegate for each of the shifted layers
    # zshape_centered = global_add_pool(
    #     hist.reshape(-1), centered_layers_perevent.reshape(-1)
    # )

    # # Estimate the turnon/turnoff width
    # get a boolean matrix to find the layers with enough hits
    occupied_layers = (
        hist > max_hits_per_event.reshape(-1, 1).repeat(1, n_layers) / 2
    )
    eventidx, occ_layer = occupied_layers.nonzero().T
    # find the min and max layer from the bool matrix ath the same time
    minmax = global_max_pool(torch.stack([occ_layer, -occ_layer]).T, eventidx)
    # reverse the -1 trick
    turnoff_layer, turnon_layer = (minmax * torch.tensor([1, -1]).to(device)).T

    # distance to the peak
    turnoff_dist = turnoff_layer - peak_layer_per_event
    turnon_dist = peak_layer_per_event - turnon_layer
    psr = (turnoff_dist + 1) / (turnon_dist + 1)

    return {
        "peak_layer": peak_layer_per_event.float(),
        "psr": psr.float(),
        "turnon_layer": turnon_layer.float(),
    }


def sphereratio(batch: Batch) -> dict[str, torch.Tensor]:
    batchidx = batch.batch
    Ehit = batch.x[:, conf.loader.x_ftx_energy_pos].reshape(-1, 1)

    e_small, e_large = __dist_fration(Ehit, batch.xyz, batchidx, 0.3, 0.8)

    return {
        "small": e_small,
        "large": e_large,
        "ratio": e_small / e_large,
    }


def cyratio(batch: Batch) -> dict[str, torch.Tensor]:
    batchidx = batch.batch
    Ehit = batch.x[:, conf.loader.x_ftx_energy_pos].reshape(-1, 1)

    e_small, e_large = __dist_fration(
        Ehit, batch.xyz[:, [0, 1]], batchidx, 0.2, 0.6
    )

    return {
        "small": e_small,
        "large": e_large,
        "ratio": e_small / e_large,
    }


def __dist_fration(Ehit, pos, batchidx, small, large, center_energy_weighted=True):
    Esum = global_add_pool(Ehit, batchidx).reshape(-1, 1)

    # get the center, weighted by energy
    if center_energy_weighted:
        center = global_add_pool(pos * Ehit, batchidx) / Esum
    else:
        center = global_mean_pool(pos, batchidx)
    std = scatter_std(pos, batchidx, dim=-2)
    # hit distance to center
    delta = (((pos - center[batchidx]) / std[batchidx]) ** 2).mean(-1).sqrt()
    del center, std
    # energy fraction inside circle around center
    e_small = (
        global_add_pool(Ehit.squeeze() * (delta < small).float(), batchidx)
        / Esum.squeeze()
    )
    e_large = (
        global_add_pool(Ehit.squeeze() * (delta < large).float(), batchidx)
        / Esum.squeeze()
    )
    # __plot_frac(e_small, e_large, small, large)
    return e_small, e_large


# def __plot_frac(e_small, e_large, small, large):
#     from matplotlib import pyplot as plt

#     ax: plt.Axes
#     fig, axes = plt.subplots(3, 1)
#     for ax, arr, title in zip(
#         axes,
#         [e_small, e_large, e_small / e_large],
#         [f"small {small}", f"large {large}", "ratio"],
#     ):
#         ax.hist(arr.cpu().numpy(), bins=100)
#         ax.set_title(title)
#     fig.tight_layout()
#     fig.savefig("wd/fig.pdf")
#     plt.close("all")


def response(batch: Batch) -> torch.Tensor:
    batchidx = batch.batch
    Ehit = batch.x[:, conf.loader.x_ftx_energy_pos]
    Esum = global_add_pool(Ehit, batchidx)

    return Esum / batch.y[:, conf.loader.y_features.index("E")]
