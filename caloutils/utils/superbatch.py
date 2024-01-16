from collections import defaultdict
from typing import Optional

import torch
import torch_scatter
from torch import LongTensor, Tensor
from torch_geometric.data import Batch
from torch_geometric.data.collate import collate


class SuperBatch(Batch):
    def __init__(self, batch_idx: LongTensor, **kwargs) -> None:
        super().__init__(**kwargs)

        if not batch_idx.dtype == torch.long:
            raise Exception("Batch index dtype must be torch.long")
        if not (batch_idx.diff() >= 0).all():
            raise Exception("Batch index must be increasing")
        if not batch_idx.dim() == 1:
            raise Exception()

        self.batch = batch_idx
        self.ptr = self._ptr_from_batchidx(batch_idx)
        self._num_graphs = int(self.batch.max() + 1)

        self._slice_dict = defaultdict(dict)
        self._inc_dict = defaultdict(dict)

    @classmethod
    def from_batch_list(
        cls,
        batches: list[Batch],
        follow_batch: Optional[list[str]] = None,
        exclude_keys: Optional[list[str]] = None,
    ):
        res, slice_dict, inc_dict = collate(
            cls,
            data_list=batches,
            increment=True,
            add_batch=not isinstance(batches[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        res._slice_dict = slice_dict
        res._inc_dict = inc_dict

        del res._slice_dict["batch"], res._inc_dict["batch"]

        res.ptr = cls._ptr_from_batchidx(cls, res.batch)

        for k in set(res.keys) - {"batch", "ptr"}:
            # slice_shift = [0] + [be._slice_dict[k][-1] for be in batches ]
            res._slice_dict[k] = pad_zero(
                torch.concat([be._slice_dict[k].diff() for be in batches]).cumsum(0)
            )
            if k != "edge_index":
                inc_shift = pad_zero(
                    torch.tensor([sum(be._inc_dict[k]) for be in batches])
                ).cumsum(0)
            else:
                inc_shift = pad_zero(
                    torch.tensor([be.num_nodes for be in batches])
                ).cumsum(0)

            res._inc_dict[k] = torch.cat(
                [
                    be._inc_dict[k] + inc_shift[ibatch]
                    for ibatch, be in enumerate(batches)
                ]
            )
        return res

    def _ptr_from_batchidx(self, batch_idx: LongTensor):
        # Construct the ptr to adress single graphs
        assert batch_idx.dtype == torch.long
        # graph[idx].x== batch.x[batch.ptr[idx]:batch.ptr[idx]+1]
        # Get delta with diff
        # Get idx of diff >0 with nonzero
        # shift by -1
        # add the batch size -1 as last element and add 0 in front
        dev = batch_idx.device
        ptr = torch.concatenate(
            (
                torch.tensor(0).long().to(dev).unsqueeze(0),
                (batch_idx.diff()).nonzero().reshape(-1) + 1,
                torch.tensor(len(batch_idx)).long().to(dev).unsqueeze(0),
            )
        )
        return ptr

    def add_node_attr(self, attrname: str, attr: Tensor):
        assert attr.device == self.batch.device
        batch_idxs = self.batch

        self[attrname] = attr
        out = torch_scatter.scatter_add(
            torch.ones(len(attr), dtype=torch.long),
            batch_idxs,
            dim=0,
        )
        out = out.cumsum(dim=0)
        self._slice_dict[attrname] = pad_zero(out)

        self._inc_dict[attrname] = torch.zeros(self._num_graphs, dtype=torch.long)

    def add_graph_attr(self, attrname: str, attr: Tensor):
        assert attr.device == self.batch.device

        self[attrname] = attr
        self._slice_dict[attrname] = torch.arange(
            self.num_graphs + 1, dtype=torch.long
        )

        self._inc_dict[attrname] = torch.zeros(self.num_graphs, dtype=torch.long)

    def set_edge_attr(self, edge_attr: LongTensor, batchidx_per_edge: LongTensor):
        assert batchidx_per_edge.dtype == torch.long
        assert (
            hasattr(self, "edge_index") and self["edge_index"].dtype == torch.long
        )
        self.edge_attr = edge_attr
        self._slice_dict["edge_attr"] = self._slice_dict["edge_index"]
        self._inc_dict["edge_attr"] = torch.zeros(self.num_graphs)

    def set_edges(self, edges: LongTensor, batchidx_per_edge: LongTensor):
        assert edges.dtype == batchidx_per_edge.dtype == torch.long
        assert edges.device == batchidx_per_edge.device == self.batch.device
        assert (
            batchidx_per_edge.diff() >= 0
        ).all(), "Edges must be ordered by batch"
        if self.edge_index is None:
            self.edge_index = torch.empty(
                2, 0, dtype=torch.long, device=self.batch.device
            )
        # Edges must be shifted by the number sum of the nodes in the previous graphs
        edges += self.ptr[batchidx_per_edge]
        self.edge_index = torch.hstack((self.edge_index.clone(), edges))
        # Fix _slice_dict
        edges_per_graph = batchidx_per_edge.unique(return_counts=True)[1]
        self._slice_dict["edge_index"] = pad_zero(edges_per_graph.cumsum(0)).cpu()
        self._inc_dict["edge_index"] = self.ptr[:-1].cpu()


def pad_zero(arr: torch.Tensor):
    return torch.cat(
        [torch.tensor(0, dtype=arr.dtype, device=arr.device).unsqueeze(0), arr]
    )
