from torch_geometric.nn import global_mean_pool, global_std_pool
from torch_geometric.data import Batch
import torch
def first_principal_components(batch: Batch) -> dict[str, torch.Tensor]:
    """
    Calculates the first principal component (PC) from a point cloud (PC).

    Parameters
    ----------
    batch : Batch
        A Batch object from the PyTorch Geometric library that contains the point cloud
        representation of the events. Batch.xyz contains the 3D coordinates of hits.
        Batch.batch contains the indices that map which hit belongs to which event.

    Returns
    -------
    fpc : dict
        Dictionary containing the components of the first principal component and the corresponding
        eigenvalue for each event. The keys of the dictionary are 'x', 'y', 'z', and 'eval'.
    """
    batchidx = batch.batch
    xyz = batch.xyz.double()
    means = global_mean_pool(xyz, batchidx)
    stds = global_std_pool(xyz, batchidx)
    deltas = (xyz - means[batchidx]) / (stds[batchidx] + 1e-8)
    cov = torch.stack(
        [
            torch.stack(
                [
                    global_mean_pool(deltas[:, i] * deltas[:, j], batchidx)
                    # global_add_pool(deltas[:, i] * deltas[:, j], batchidx)
                    # / (batch.ptr[1:]-batch.ptr[:-1] - 1)  # normalize
                    for i in range(3)
                ]
            )
            for j in range(3)
        ]
    ).transpose(0, -1)

    e_vals, e_vec = torch.linalg.eigh(cov)

    # largest_ev = e_val.argmax(-1).reshape(-1, 1, 1)
    # first_pc = e_vec.take_along_dim(largest_ev, -1)
    # (first_pc==e_vec[:,:,[2]]).all() why?? are they sorted?
    # https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html
    # e_vec are sorted, get last colomn
    first_pc = e_vec[:, :, -1]

    # TEST
    # untrfs = deltas[batchidx==0]
    # first pc in the the first component:
    # fct = (first_pc[0].reshape(1, 3) @ untrfs.T).reshape(-1, 1) * first_pc[
    #     0
    # ].reshape(1, 3)
    # assert ((untrfs-fct).std(0)<(untrfs).std(0)).all()
    return dict(zip(["x", "y", "z"], first_pc.T.float().abs())) | {
        "eval": e_vals[:, -1].float()
    }