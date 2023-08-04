import torch
from torch_scatter import scatter_add

from caloutils.processing import sum_multi_hits


def test_sum_multi_hits():
    from torch_geometric.data import Batch, Data

    batch = Batch.from_data_list(
        [
            Data(
                x=torch.tensor(
                    [
                        [1, 0, 0, 0],
                    ]
                ),
                y=torch.tensor([[1, 1]]),
                # n_pointsv=torch.tensor(3),
            ),
            Data(
                x=torch.tensor(
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 1, 1, 0],
                        [1, 1, 1, 0],
                        # [1, 1, 0, 0],
                        # [1, 1, 0, 0],
                        # [1, 0, 0, 0],
                        # [1, num_z - 1, num_alpha - 1, num_r - 1],
                    ]
                ),
                y=torch.tensor([[1, 1]]),
                # n_pointsv=torch.tensor(4),
            ),
        ]
    )
    batch_new = sum_multi_hits(batch.clone())

    batchidx = batch.batch
    batchidx_new = batch_new.batch

    n_multihit = batch_new.n_multihit
    new_counts = batch_new.n_pointsv

    hitE = batch.x[:, 0]
    pos = batch.x[:, 1:].long()
    hitE_new = batch_new.x[:, 0]
    pos_new = batch_new.x[:, 1:].long()

    old_counts = torch.unique_consecutive(batchidx, return_counts=True)[1]
    if "n_pointsv" in batch.keys:
        assert (old_counts == batch.n_pointsv).all()
    assert ((old_counts - new_counts) == n_multihit).all()
    assert torch.allclose(
        scatter_add(hitE_new, batchidx_new), scatter_add(hitE, batchidx)
    )
    assert torch.allclose(
        scatter_add(pos_new * hitE_new.unsqueeze(-1), batchidx_new, -2),
        scatter_add(pos * hitE.unsqueeze(-1), batchidx, -2),
    )
