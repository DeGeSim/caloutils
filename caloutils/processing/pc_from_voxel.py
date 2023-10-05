import torch
from torch import Tensor
from torch_geometric.data import Batch

from ..utils.batch import add_graph_attr, add_node_attr, init_batch


def voxel_to_pc(voxel: Tensor, Einc: Tensor) -> Batch:
    """Converts a 3D voxel tensor representation of a particle shower into a hit cloud (pc).
    The conversion is done by finding all non-zero elements in the voxel tensor,
    storing their values and their coordinates in 3D space, and grouping them by shower.

    Parameters
    ----------
    voxel : Tensor
        A 4D tensor representing multiple particle showers in pc_to_voxeld format.
        Dimensions are (num_showers, num_z, num_alpha, num_r).
    Einc : Tensor
        A 1D tensor representing containing the incoming energy of each shower.
        Dimensions are (num_showers,).

    Returns
    -------
    batch: Batch
        A Batch object from the PyTorch Geometric library that contains the point cloud
        representation of the showers, where:
        - batch.batch is a 1D tensor where each element is the ID of the shower the corresponding
        point in the point cloud belongs to.
        - batch.x is a 3D tensor where each row represents a hit in the point cloud,
        with the columns being the value of the voxel and its 3D coordinates (r, alpha, z).
        - batch.y is a 2D tensor where each row represents the incoming energy of the shower
        and the number of non-zero voxels (hits) in it.

    Raises
    ------
    AssertionError
        If there are NaN values in the point cloud tensor x.
    """
    E, showers = Einc.clone(), voxel.clone()
    assert E.dim() == 1
    assert E.dim() < voxel.dim()
    assert (
        E.shape[0] == voxel.shape[0]
    ), "Both inputs must give the same number of events"

    coords = torch.argwhere(
        showers > 0.0
    )  # get indices of non-zero values (shower_id, r, alpha, z)
    vals = showers[tuple(coords.T)]  # get non-zero values
    _, num_hits = torch.unique(
        coords[:, 0], return_counts=True
    )  # get number of non-zero values per shower
    coords = coords[:, 1:]  # remove shower_id from coords
    start_index = torch.zeros(
        num_hits.shape, dtype=torch.int64
    )  # create start_index array
    start_index[1:] = num_hits.cumsum(0)[:-1]  # calculate start_index
    x = torch.hstack((vals[:, None], coords))
    y = torch.stack([E, num_hits], dim=-1)

    # create batch_idx array
    batch_idx = torch.arange(showers.shape[0]).repeat_interleave(num_hits).long()
    batch = init_batch(batch_idx)
    add_node_attr(batch, "x", x)
    add_graph_attr(batch, "y", y)

    return batch
