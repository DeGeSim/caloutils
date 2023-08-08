from typing import Optional

import torch
from torch_geometric.data import Batch

from .. import calorimeter
from .utils import scatter_sort


def shift_multi_hits(
    batch: Batch,
    globalidx: Optional[torch.Tensor] = None,
    return_globalidx: bool = False,
):
    """
    Shifts hits that belong to the same cell, to neighboring, empty cells, if available.
    Leaves the highest energy hit in the cell and prioritizes moving the 2nd most energetic (then 3rd most ...) hit.
    The function modifies the batch in place, updating batch.x, batch.batch, and batch.ptr.

    Parameters
    ----------
    batch : Batch
        A Batch object from the PyTorch Geometric library that contains the point cloud
        representation of the events. Batch.x contains the hit energy and 3D coordinates of hits.
        Batch.batch contains the indices that map which hit belongs to which shower.

    forbid_dublicates : bool, optional
        If True, asserts that there were no duplicate hits in the original data. Defaults to True.

    Returns
    -------
    batch : Batch
        The modified Batch object where duplicate hits have been summed.
    """
    batchidx: torch.Tensor = batch.batch
    dev = batch.x.device

    if globalidx is None:
        globalidx = calorimeter.globalidx_from_pos(batch.x[:, 1:].long(), batchidx)

    # sort, so we can check globalidx for repetitions to check for double cells
    globalidx, index_perm = scatter_sort(globalidx, batchidx)
    batch.x = batch.x[index_perm]

    # The idea of this algorithm is to look for repetitions
    # in the (sorted) globalidx to detect repetitions.

    global_uniques, unique_cells_idx, counts = torch.unique(
        globalidx, return_inverse=True, return_counts=True
    )

    # Because we want to keep the highest energy hit in the same
    # cell we now sort the hits in the same cell by the energy
    # For this, we need to know which hits share a cell
    has_multihits = torch.isin(globalidx, global_uniques[counts > 1])
    # Here we want to sort energies of the same cell/event,
    # so we use the globalidx instead of the usual batchidx
    _, perm = scatter_sort(
        batch.x[..., 0][has_multihits], globalidx[has_multihits], descending=True
    )
    batch.x[has_multihits] = batch.x[has_multihits][perm]
    globalidx[has_multihits] = globalidx[has_multihits][perm]
    assert (batchidx[has_multihits] == batchidx[has_multihits][perm]).all()

    # select the second hit in each cell for moving around
    mhit_idxs = (
        torch.cat([torch.tensor([0]).to(dev), 1 - unique_cells_idx.diff()])
        .nonzero()
        .squeeze()
    )
    shift_options = len(calorimeter.dims) * 2
    # start value for each shift
    shift_state = torch.randint_like(mhit_idxs, 1, shift_options)

    # switch for testing this function
    if calorimeter._caloname == "test":
        shift_state = torch.ones_like(shift_state)
    idxs_to_overwrite = []
    new_x_list = []
    new_global_list = []

    for _ in range(shift_options):
        if len(mhit_idxs) == 0:
            break

        batchidx_to_shift = batchidx[mhit_idxs]
        new_x = _shift_pos(batch.x[mhit_idxs], shift_state)

        # # sort by highest energy, to prioritize shifting
        # # the 2nd highest energy out of the cell
        # _, perm = scatter_sort(new_x.T[0], batchidx_to_shift, descending=True)
        # new_x = new_x[perm]
        # mhit_idxs = mhit_idxs[perm]

        # get the new globals index for the shifted hits
        new_global = calorimeter.globalidx_from_pos(new_x[:, 1:], batchidx_to_shift)

        # Check if the shift is valid
        # A) Check if this is the first hit to be shifted to this cell
        # we now need and index that filters mhit_idxs
        # and only lets the first though.
        # (4,3,4,1,3)   -> (1,1,0,1,0) or (0,1,3)
        # we can check if the index  of new_global.unique is increaseing
        # but for this we need to sort the indexes again
        new_global, perm = new_global.sort()
        mhit_idxs = mhit_idxs[perm]
        new_x = new_x[perm]

        # now we can check of the unique rev index is increasing
        # which tells us of the cell is being accessed for the first time

        first_new_hit = torch.cat(
            (
                torch.tensor([1]).to(dev),
                new_global.unique(return_inverse=True)[1].diff(),
            )
        ).bool()

        # B)
        # check if the new position is valid iff it's not already occupied
        new_pos_is_free = ~torch.isin(
            new_global, torch.cat([globalidx, *new_global_list])
        )

        # combine the two
        valide_shift_index = new_pos_is_free & first_new_hit

        # apppend the shifted hits to the list
        if valide_shift_index.sum():
            idxs_to_overwrite.append(mhit_idxs[valide_shift_index])
            new_x_list.append(new_x[valide_shift_index])
            new_global_list.append(new_global[valide_shift_index])
        # rerun loop with the remaining multihits
        mhit_idxs = mhit_idxs[~valide_shift_index]
        shift_state = (shift_state[~valide_shift_index] + 1) % shift_options

    if len(idxs_to_overwrite):
        # overwrite old position and global index
        stacked_idxs_to_overwrite = torch.cat(idxs_to_overwrite)
        globalidx[stacked_idxs_to_overwrite] = torch.cat(new_global_list)
        batch.x[stacked_idxs_to_overwrite] = torch.cat(new_x_list)

    if return_globalidx:
        return batch, globalidx
    else:
        return batch


def _shift_pos(x, shift_state):
    pos = x[:, 1:].long()
    dev = pos.device
    directon = (shift_state % 2) * 2 - 1
    dim = shift_state // 2
    pos[torch.arange(len(pos)), dim] += directon

    # rotate alpha
    num_alpha = calorimeter.num_alpha
    alphas = pos[:, 1]
    alphas[alphas > num_alpha - 1] -= num_alpha
    alphas[alphas < 0] -= num_alpha
    pos[:, 1] = alphas

    # clamp r and z
    pos = torch.clamp(
        pos,
        torch.tensor([0, 0, 0]).to(dev),
        torch.tensor(calorimeter.dims).to(dev) - 1,
    )
    x[:, 1:] = pos
    return x
