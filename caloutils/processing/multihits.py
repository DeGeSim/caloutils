import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_add

from .. import calorimeter
from .shift_hits import _shift_multihits_to_neighbor_cells
from .utils import fix_slice_dict_nodeattr, ptr_from_batchidx, scatter_sort


def sum_multi_hits(batch, forbid_dublicates=False, shiftmultihit=False):
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

    shiftmultihit : bool, optional
        If True, tries to move hits from overfilled cells to neighboring empty cells. Defaults to True.

    Returns
    -------
    batch : Batch
        The modified Batch object where duplicate hits have been summed.
    """

    # batch = batch.to("cpu")
    batchidx = batch.batch
    assert (batchidx.diff() >= 0).all()
    globalidx = calorimeter.globalidx_from_pos(batch.x[:, 1:].long(), batchidx)

    if shiftmultihit:
        # get new positions and global index
        # and the current index of these events
        batch, globalidx = _shift_multihits_to_neighbor_cells(
            batch, globalidx.clone(), return_globalidx=True
        )

    batch = _add_hits(batch, globalidx, forbid_dublicates)
    return batch


def _add_hits(batch: Batch, globalidx: torch.Tensor, forbid_dublicates: bool):
    dev = batch.x.device
    batchidx = batch.batch

    # sort the globalidx
    globalidx, index_perm = scatter_sort(globalidx, batchidx)
    batch.x = batch.x[index_perm]

    hitE = batch.x[:, 0]
    pos = batch.x[:, 1:].long()
    assert (batchidx[index_perm] == batchidx).all()

    # unique_cells_idx counts up every time a new cell in an
    # even is accessed in globalidx
    # counts gives the times the cell/event idx is occupied
    _, unique_cells_idx, counts = torch.unique(
        globalidx, return_inverse=True, return_counts=True
    )

    if forbid_dublicates:
        assert (counts - 1 == 0).all()

    # begin sum
    hitE_new = scatter_add(hitE, unique_cells_idx)
    sel_new_idx = counts.cumsum(-1) - 1
    if forbid_dublicates:
        assert (sel_new_idx == torch.arange(len(batch.x)).to(dev)).all()

    batchidx_new = batchidx[sel_new_idx]
    pos_new = pos[sel_new_idx]

    # count the cells, that have been hit multiple times
    n_multihit = scatter_add(counts - 1, batchidx_new)
    if forbid_dublicates:
        assert (n_multihit == 0).all()
    new_counts = torch.unique_consecutive(batchidx_new, return_counts=True)[1]

    x_new = torch.hstack([hitE_new.reshape(-1, 1), pos_new])

    # TODO remove sanity test:
    old_counts = torch.unique_consecutive(batchidx, return_counts=True)[1]
    if "n_pointsv" in batch.keys:
        assert (old_counts == batch.n_pointsv).all()
    assert ((old_counts - new_counts) == n_multihit).all()
    assert torch.allclose(
        scatter_add(hitE_new, batchidx_new), scatter_add(hitE, batchidx)
    )
    # if not shiftmultihit:
    #     assert torch.allclose(
    #         scatter_add(pos_new * hitE_new.unsqueeze(-1), batchidx_new, -2),
    #         scatter_add(pos * hitE.unsqueeze(-1), batchidx, -2),
    #     )

    if forbid_dublicates:
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

    return batch.to(dev)
