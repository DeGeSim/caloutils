from collections import defaultdict

import torch
import torch_scatter
from torch import LongTensor, Tensor
from torch_geometric.data import Batch


def init_batch(batch_idx: LongTensor):
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


def ptr_from_batchidx(batch_idx: LongTensor):
    # Construct the ptr to adress single graphs
    assert batch_idx.dtype == torch.long
    # graph[idx].x== batch.x[batch.ptr[idx]:batch.ptr[idx]+1]
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


def add_node_attr(batch: Batch, attrname: str, attr: Tensor):
    device = attr.device
    assert device == batch.batch.device
    batch_idxs = batch.batch

    batch[attrname] = attr
    out = torch_scatter.scatter_add(
        torch.ones(len(attr), dtype=torch.long, device=device), batch_idxs, dim=0
    )
    out = out.cumsum(dim=0)
    # batch._slice_dict[attrname] =  torch.cat(
    #     [torch.zeros(1, dtype=torch.long, device=device), out], dim=0
    # )
    batch._slice_dict[attrname] = pad_zero(out).cpu()

    batch._inc_dict[attrname] = torch.zeros(
        batch._num_graphs, dtype=torch.long, device=device
    ).cpu()


def add_graph_attr(batch: Batch, attrname: str, attr: Tensor):
    device = attr.device
    assert device == batch.batch.device

    batch[attrname] = attr
    batch._slice_dict[attrname] = torch.arange(
        batch.num_graphs + 1, dtype=torch.long, device=device
    ).cpu()

    batch._inc_dict[attrname] = torch.zeros(
        batch.num_graphs, dtype=torch.long, device=device
    ).cpu()


def set_edge_attr(
    batch: Batch, edge_attr: LongTensor, batchidx_per_edge: LongTensor
):
    assert batchidx_per_edge.dtype == torch.long
    assert hasattr(batch, "edge_index") and batch["edge_index"].dtype == torch.long
    batch.edge_attr = edge_attr
    # Fix _slice_dict
    # edges_per_graph = batchidx_per_edge.unique(return_counts=True)[1]
    # batch._slice_dict["edge_attr"] = pad_zero(edges_per_graph.cumsum(0))
    batch._slice_dict["edge_attr"] = batch._slice_dict["edge_index"].cpu()
    batch._inc_dict["edge_attr"] = torch.zeros(
        batch.num_graphs, device=edge_attr.device
    ).cpu()


def set_edges(batch: Batch, edges: LongTensor, batchidx_per_edge: LongTensor):
    assert edges.dtype == batchidx_per_edge.dtype == torch.long
    assert edges.device == batchidx_per_edge.device == batch.batch.device
    assert (batchidx_per_edge.diff() >= 0).all(), "Edges must be ordered by batch"
    if batch.edge_index is None:
        batch.edge_index = torch.empty(
            2, 0, dtype=torch.long, device=batch.batch.device
        )
    # Edges must be shifted by the number sum of the nodes in the previous graphs
    edges += batch.ptr[batchidx_per_edge]
    batch.edge_index = torch.hstack((batch.edge_index.clone(), edges))
    # Fix _slice_dict
    edges_per_graph = batchidx_per_edge.unique(return_counts=True)[1]
    batch._slice_dict["edge_index"] = pad_zero(edges_per_graph.cumsum(0)).cpu()
    batch._inc_dict["edge_index"] = batch.ptr[:-1].cpu()


def pad_zero(arr: torch.Tensor):
    return torch.cat(
        [torch.tensor(0, dtype=arr.dtype, device=arr.device).unsqueeze(0), arr]
    )


def pad_one(arr: torch.Tensor):
    return torch.cat(
        [torch.tensor(1, dtype=arr.dtype, device=arr.device).unsqueeze(0), arr]
    )


def from_batch_list(*batches: list[Batch]):
    res = Batch.from_data_list(batches)
    res.ptr = ptr_from_batchidx(res.batch)
    k = "edge_attr"
    for k in res.keys:
        res._slice_dict[k] = torch.cat(
            [
                be._slice_dict[k]
                + (0 if ibatch == 0 else batches[ibatch - 1]._slice_dict[k][-1])
                for ibatch, be in enumerate(batches)
            ]
        )
        # res._inc_dict[k] = torch.cat(
        #     [
        #         be._slice_dict[k]
        #         for ibatch, be in enumerate(batches)
        #     ]
        # )
    return res
