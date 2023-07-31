import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool


def response(batch: Batch) -> torch.Tensor:
    """
    Computes the energy response for each event in a batch.

    The energy response is calculated as the ratio of the total detected energy
    (sum of energy from all hits in an event) to the true energy of the event.
    Parameters
    ----------
    batch : Batch
        A Batch object from the PyTorch Geometric library that contains the point cloud
        representation of the events.

    Returns
    -------
    torch.Tensor
        A tensor of the energy response for each event in the batch.
    """

    batchidx = batch.batch
    Ehit = batch.x[:, 0]
    Esum = global_add_pool(Ehit, batchidx)
    return Esum / batch.y[:, 0]
