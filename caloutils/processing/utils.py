from collections import defaultdict
from math import prod

import torch
import torch_scatter
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import pool
from torch_scatter import scatter_add

from .. import calorimeter


def _scatter_sort(x: Tensor, index: Tensor, dim=-1):
    """
    Sorts the elements of `x` that share the same `index`, returns sorted `x` and `index` needed.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to sort.
    index : torch.Tensor
        The index tensor to sort according to `x`.
    dim : int, optional
        The dimension along which to sort. Defaults to the last dimension.

    Returns
    -------
    x : torch.Tensor
        The sorted input tensor.
    index : torch.Tensor
        The index tensor sorted according to `x`.
    """
    x, x_perm = torch.sort(x, dim=dim)
    index = index.take_along_dim(x_perm, dim=dim)
    index, index_perm = torch.sort(index, dim=dim, stable=True)
    x = x.take_along_dim(index_perm, dim=dim)
    return x, x_perm.take_along_dim(index_perm, dim=dim)


def sum_duplicate_hits(batch: Batch, fake=True):
    """
    Sums the energy of duplicate hits in the same cell for each event.
    If fake=False, the function also verifies that there were no duplicate hits in the original data.
    The function modifies the batch in place, updating batch.x, batch.batch, and batch.ptr.

    Parameters
    ----------
    batch : Batch
        A Batch object from the PyTorch Geometric library that contains the point cloud
        representation of the events. Batch.x contains the hit energy and 3D coordinates of hits.
        Batch.batch contains the indices that map which hit belongs to which shower.

    fake : bool, optional
        If True, asserts that there were no duplicate hits in the original data. Defaults to True.

    Returns
    -------
    batch : Batch
        The modified Batch object where duplicate hits have been summed.
    """
    old_dev = batch.x.device
    batch = batch.to("cpu")
    dev = batch.x.device
    batchidx = batch.batch
    n_events = int(batchidx[-1] + 1)

    hitE = batch.x[:, 0]
    pos = batch.x[:, 1:].long()

    pos_z, pos_alpha, pos_r = pos.T
    assert (pos >= 0).all()
    assert (pos_z < calorimeter.num_z).all()
    assert (pos_alpha < calorimeter.num_alpha).all()
    assert (pos_r < calorimeter.num_r).all()

    cell_idxs = torch.arange(
        prod((calorimeter.num_z, calorimeter.num_alpha, calorimeter.num_r))
    ).reshape(*(calorimeter.num_z, calorimeter.num_alpha, calorimeter.num_r))
    cell_idx_per_hit = cell_idxs.to(dev)[pos_z, pos_alpha, pos_r]

    # give pos unique values for each event
    eventshift = torch.arange(n_events).to(dev) * prod(calorimeter.dims)
    event_and_cell_idxs = cell_idx_per_hit + eventshift[batchidx]

    # sort the event_and_cell_idxs
    cell_idx_per_hit, index_perm = _scatter_sort(event_and_cell_idxs, batchidx)
    hitE = hitE[index_perm]
    pos = pos[index_perm]
    assert (batchidx[index_perm] == batchidx).all()

    _, invidx, counts = torch.unique_consecutive(
        cell_idx_per_hit, return_inverse=True, return_counts=True
    )
    if not fake:
        assert (counts - 1 == 0).all()

    hitE_new = scatter_add(hitE, invidx)
    sel_new_idx = counts.cumsum(-1) - 1
    if not fake:
        assert (sel_new_idx == torch.arange(len(batch.x))).all()

    batchidx_new = batchidx[sel_new_idx]
    pos_new = pos[sel_new_idx]

    # count the cells, that have been hit multiple times
    n_multihit = scatter_add(counts - 1, batchidx_new)
    if not fake:
        assert (n_multihit == 0).all()
    new_counts = torch.unique_consecutive(batchidx_new, return_counts=True)[1]

    x_new = torch.hstack([hitE_new.reshape(-1, 1), pos_new])

    # # Tests
    # old_counts = torch.unique_consecutive(batchidx, return_counts=True)[1]
    # if "n_pointsv" in batch.keys:
    #     assert (old_counts == batch.n_pointsv).all()
    # assert ((old_counts - new_counts) == n_multihit).all()
    # assert torch.allclose(
    #     scatter_add(hitE_new, batchidx_new), scatter_add(hitE, batchidx)
    # )
    # assert torch.allclose(
    #     scatter_add(pos_new * hitE_new.unsqueeze(-1), batchidx_new, -2),
    #     scatter_add(pos * hitE.unsqueeze(-1), batchidx, -2),
    # )
    if not fake:
        assert (n_multihit == 0).all()
        assert (batch.batch == batchidx_new).all()
        assert (batch.n_pointsv == new_counts).all()
        for i in range(4):
            assert torch.allclose(batch.x.T[i][index_perm], x_new.T[i])

    batch.n_multihit = n_multihit
    batch.batch = batchidx_new
    batch.x = x_new
    batch.n_pointsv = new_counts
    # need to shift the ptr by the number of removed hits
    batch.ptr = ptr_from_batchidx(batchidx_new)

    batch.nhits = {
        "n": batch.n_pointsv,
        "n_by_E": batch.n_pointsv / batch.y[:, 0],
    }
    fix_slice_dict_nodeattr(batch, "x")

    return batch.to(old_dev)


def ptr_from_batchidx(batch_idx: Tensor):
    """
    Constructs a pointer tensor from the given batch index tensor. The pointer tensor allows to address
    individual graphs within a batch of graphs.

    The constructed pointer tensor corresponds to the cumulative sum of the number of nodes in each individual graph.
    This allows to index specific graphs within the batch.

    Parameters
    ----------
    batch_idx : torch.Tensor
        1D tensor containing the batch indices for each node in the batch.
        Each value in batch_idx indicates which graph the node belongs to.
        For example, if batch_idx[i] = j, then the ith node belongs to the jth graph in the batch.

    Returns
    -------
    ptr : torch.Tensor
        Pointer tensor constructed from the batch_idx. This tensor can be used to index individual graphs in the batch.
        For example, if ptr[k] = m, then the first node of the kth graph in the batch is the mth node in the node feature matrix.

    Examples
    --------
    >>> batch_idx = torch.tensor([0, 0, 1, 1, 1, 2])
    >>> ptr_from_batchidx(batch_idx)
    tensor([0, 2, 5, 6])
    """
    dev = batch_idx.device
    return torch.concatenate(
        (
            torch.tensor(0).long().to(dev).unsqueeze(0),
            (batch_idx.diff()).nonzero().reshape(-1) + 1,
            torch.tensor(len(batch_idx)).long().to(dev).unsqueeze(0),
        )
    )


def fix_slice_dict_nodeattr(batch: Batch, attrname: str) -> Batch:
    """
    Modifies the "_slice_dict" attribute of the Batch object to include the number of elements of a given attribute
    per graph in the batch.

    Parameters
    ----------
    batch : Batch
        The batch of graphs.
    attrname : str
        The attribute of the batch for which the slice dictionary is to be updated.

    Returns
    -------
    batch : Batch
        The modified batch object with updated "_slice_dict".

    Note
    ----
    The "_slice_dict" attribute of the batch object is a dictionary which stores the number of elements
    of various attributes per graph in the batch. This function updates this dictionary for a given attribute.
    """
    if not hasattr(batch, "_slice_dict"):
        batch._slice_dict = defaultdict(dict)
    attr = batch[attrname]
    batch_idxs = batch.batch
    device = attr.device
    out = torch_scatter.scatter_add(
        torch.ones(len(attr), dtype=torch.long, device=device), batch_idxs, dim=0
    )
    out = out.cumsum(dim=0)
    batch._slice_dict[attrname] = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), out], dim=0
    )
    return batch


def global_std_pool(x: Tensor, batchidx: Tensor) -> Tensor:
    """
    Computes the global standard deviation of node features in a batch of graphs.

    Parameters
    ----------
    x : Tensor
        The node features of the batch.
    batchidx : Tensor
        The batch indices for each node in the batch.

    Returns
    -------
    Tensor
        A tensor containing the global standard deviation of node features for each graph in the batch.
    """
    return torch.sqrt(global_var_pool(x, batchidx))


def global_var_pool(x: Tensor, batchidx: Tensor) -> Tensor:
    """
    Computes the global variance of node features in a batch of graphs.

    Parameters
    ----------
    x : Tensor
        The node features of the batch.
    batchidx : Tensor
        The batch indices for each node in the batch.

    Returns
    -------
    Tensor
        A tensor containing the global variance of node features for each graph in the batch.
    """
    means = pool.global_mean_pool(x, batchidx)
    deltas = torch.pow(means[batchidx] - x, 2)
    return pool.global_mean_pool(deltas, batchidx)
