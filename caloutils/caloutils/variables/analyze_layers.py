from torch_geometric.data import Batch
from torch_geometric.nn.pool import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch import Tensor
import torch
from ..caloutils import calorimeter

def analyze_layers(batch: Batch) -> dict[str, Tensor]:
    """
    Analyzes the layers of a point cloud representation of particle showers, computing
    the peak layer, the turn-on layer, and the ratio of distances from turn-off layer to
    the peak layer and from the peak layer to the turn-on layer.

    The function performs the following main steps:
    1. Computes a histogram of the z coordinates (layers) of each point in the point cloud.
    2. Identifies the layer with the maximum number of points (peak layer) for each shower.
    3. Identifies the first (turn-on) and last (turn-off) layer that contains at least half
    the maximum number of points in a layer for each shower.
    4. Computes the ratio of the distance from the turn-off layer to the peak layer and
    the distance from the peak layer to the turn-on layer.

    Parameters
    ----------
    batch : Batch
        A Batch object from the PyTorch Geometric library that contains the point cloud
        representation of the showers.

    Returns
    -------
    dict
        A dictionary with keys 'peak_layer', 'psr', and 'turnon_layer', where:
            - 'peak_layer' corresponds to the tensor of peak layers for each shower in the batch.
            - 'psr' corresponds to the tensor of ratios for each shower in the batch.
            - 'turnon_layer' corresponds to the tensor of turn-on layers for each shower in the batch.

    Raises
    ------
    AssertionError
        If the sum of the elements in each row of the histogram does not equal
        the difference of consecutive elements in the batch pointers.

    """
    batchidx = batch.batch
    n_events = int(batchidx[-1] + 1)
    device = batch.x.device
    z = batch.xyz[..., -1]
    # compute the histogramm with dim (events,layers)
    z_ev = (z.long() + batchidx * calorimeter.num_z).sort().values

    zvals, zcounts = z_ev.unique_consecutive(return_counts=True)

    layer_idx = zvals % calorimeter.num_z
    ev_idx = (zvals - layer_idx) // calorimeter.num_z

    hist = torch.zeros(n_events, calorimeter.num_z).long().to(device)
    hist[ev_idx, layer_idx] = zcounts

    assert (hist.sum(1) == batch.ptr[1:] - batch.ptr[:-1]).all()

    max_hits_per_event, peak_layer_per_event = hist.max(1)

    occupied_layers = (
        hist > max_hits_per_event.reshape(-1, 1).repeat(1, calorimeter.num_z) / 2
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
