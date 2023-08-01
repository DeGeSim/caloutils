from math import prod

import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_add

from .. import calorimeter


def voxelize(batch: Batch) -> torch.Tensor:
    """
    Converts a pytorch geometric batch of point clouds into a torch batch of 3D voxel grids.

    Parameters
    ----------
    batch : Batch
        A Batch object from the PyTorch Geometric library that contains the point cloud
        representation of the events.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, num_z, num_alpha, num_r), where each element
        represents the energy in the corresponding voxel.
    """
    dims = calorimeter.dims
    batch_size = int(batch.batch[-1] + 1)
    x = batch.x
    shower_index = batch.batch
    Ehit = x.T[0]
    valid_coordinates = x.T[1:].int()
    indices = torch.cat((shower_index.unsqueeze(1), valid_coordinates.t()), dim=1)

    full_event_cell_idx = (
        torch.arange(batch_size * dims[0] * dims[1] * dims[2])
        .reshape(batch_size, *dims)
        .to(x.device)
    )
    scatter_index = full_event_cell_idx[
        indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]
    ]
    vox = scatter_add(
        src=Ehit,
        index=scatter_index,
        dim_size=prod(dims) * batch_size,
    )
    return vox.reshape(batch_size, *dims)
