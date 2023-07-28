from collections import defaultdict

import torch
import torch_scatter
from torch_geometric.data import Batch


def init_batch(batch_idx: torch.Tensor):
    if not batch_idx.dtype == torch.long:
        raise Exception("Batch index dtype must be torch.long")
    if not (batch_idx.diff() >= 0).all():
        raise Exception("Batch index must be increasing")
    if not batch_idx.dim() == 1:
        raise Exception()

    batch = Batch(batch=batch_idx)

    batch.ptr = ptr_from_batchidx(batch_idx)
    batch._num_graphs = int(batch.batch.max() + 1)

    batch._slice_dict = defaultdict(dict)
    batch._inc_dict = defaultdict(dict)
    return batch


def ptr_from_batchidx(batch_idx):
    # Construct the ptr to adress single graphs
    # graph[idx].x= batch.x[batch.ptr[idx]:batch.ptr[idx]+1]
    # Get delta with diff
    # Get idx of diff >0 with nonzero
    # shift by -1
    # add the batch size -1 as last element and add 0 in front
    dev = batch_idx.device
    return torch.concatenate(
        (
            torch.tensor(0).long().to(dev).unsqueeze(0),
            (batch_idx.diff()).nonzero().reshape(-1) + 1,
            torch.tensor(len(batch_idx)).long().to(dev).unsqueeze(0),
        )
    )


def add_nodewise_attr(batch: Batch, attrname: str, attr: torch.Tensor):
    device = attr.device
    assert device == batch.batch.device
    batch_idxs = batch.batch

    batch[attrname] = attr
    out = torch_scatter.scatter_add(
        torch.ones(len(attr), dtype=torch.long, device=device), batch_idxs, dim=0
    )
    out = out.cumsum(dim=0)
    batch._slice_dict[attrname] = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), out], dim=0
    )

    batch._inc_dict[attrname] = torch.zeros(
        batch._num_graphs, dtype=torch.long, device=device
    )
