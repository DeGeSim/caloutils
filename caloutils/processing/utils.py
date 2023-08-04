from collections import defaultdict

import torch
import torch_scatter
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import pool

from .. import calorimeter


def scatter_sort(x: Tensor, index: Tensor, dim=-1):
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


def _construct_global_cellevent_idx(batch_size: int) -> torch.Tensor:
    dims = calorimeter.dims
    full_event_cell_idx = torch.arange(
        batch_size * dims[0] * dims[1] * dims[2]
    ).reshape(batch_size, *dims)
    return full_event_cell_idx
