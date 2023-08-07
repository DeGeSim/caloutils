import torch
from torch_scatter import scatter_add

import caloutils
from caloutils.processing import shift_hits

shift_hits._testing_no_random_shift = True

caloutils.init_calorimeter("test")


def test_shift_multihits_to_neighbor_cells():
    from torch_geometric.data import Batch, Data

    batch = Batch.from_data_list(
        [
            Data(
                x=torch.tensor(
                    [
                        [7, 0, 0, 0],
                    ]
                ),
                y=torch.tensor([[1, 1]]),
            ),
            Data(
                x=torch.tensor(
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [3, 0, 0, 0],
                        [1, 0, 0, 0],
                    ]
                ),
                y=torch.tensor([[1, 1]]),
            ),
        ]
    )
    batch_new = shift_hits._shift_multihits_to_neighbor_cells(batch.clone())

    assert (
        batch_new.x
        == torch.tensor(
            [
                [7, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [3, 1, 0, 0],
                [1, 0, 0, 0],
            ]
        )
    ).all()

    batchidx = batch.batch
    hitE = batch.x[:, 0]
    hitE_new = batch_new.x[:, 0]

    old_counts = torch.unique_consecutive(batchidx, return_counts=True)[1]
    if "n_pointsv" in batch.keys:
        assert (old_counts == batch.n_pointsv).all()
    assert ((old_counts - batch_new.n_pointsv) == batch_new.n_multihit).all()
    assert torch.allclose(
        scatter_add(hitE_new, batch_new.batch), scatter_add(hitE, batchidx)
    )
