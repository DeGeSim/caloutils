import torch
from torch_scatter import scatter_add

import caloutils
from caloutils.processing import shift_multi_hits

caloutils.init_calorimeter("test")


def testshift_multi_hits():
    from torch_geometric.data import Batch, Data

    batch = Batch.from_data_list(
        [
            Data(x=torch.tensor([[7, 0, 0, 0]])),
            Data(
                x=torch.tensor(
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [3, 0, 0, 0],
                        [1, 0, 0, 0],
                    ]
                )
            ),
            Data(x=torch.tensor([[1, 1, 1, 0], [2, 1, 1, 0]])),
        ]
    )
    batch_new = shift_multi_hits(batch.clone())

    assert (
        batch_new.x
        == torch.tensor(
            [
                [7, 0, 0, 0],
                [3, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 0],
                [2, 1, 1, 0],
                [1, 1, 0, 0],
            ]
        )
    ).all()

    batchidx = batch.batch

    old_counts = torch.unique_consecutive(batchidx, return_counts=True)[1]
    if "n_pointsv" in batch.keys:
        assert (old_counts == batch.n_pointsv).all()
        assert ((old_counts - batch_new.n_pointsv) == batch_new.n_multihit).all()
    assert torch.allclose(
        scatter_add(batch_new.x[:, 0], batch_new.batch),
        scatter_add(batch.x[:, 0], batchidx),
    )
