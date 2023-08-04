from math import prod

import torch
from torch_scatter import scatter_add

from .. import calorimeter
from .utils import _construct_global_cellevent_idx


def nhits_in_same_cell(batch):
    batch_size = int(batch.batch[-1] + 1)
    dev = batch.x.device
    fulldim = (batch_size, *calorimeter.dims)
    full_event_cell_idx = _construct_global_cellevent_idx(batch_size).to(dev)

    indices = torch.hstack((batch.batch.unsqueeze(1), batch.pos))
    scatter_index = full_event_cell_idx[
        indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]
    ]
    occ = scatter_add(
        src=torch.ones(len(batch.batch), dtype=torch.int, device=dev),
        index=scatter_index,
        dim_size=prod(calorimeter.dims) * batch_size,
    ).reshape(*fulldim)
    batch_occ = occ[
        indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]
    ]
    return batch_occ
