import torch
from torch_geometric.data import Batch, Data

from caloutils.utils.batch import add_nodewise_attr, init_batch


def test_batch_utils():
    batch_size = 3
    node_range = (1, 10)
    nodes_v = torch.randint(*node_range, (batch_size,))
    x_list = [torch.rand(n, 3) for n in nodes_v]

    batch_list = [Data(x=x) for x in x_list]
    batch_truth = Batch.from_data_list(batch_list)

    batch_idx = batch_truth.batch

    batch = init_batch(batch_idx)

    add_nodewise_attr(batch, "x", torch.vstack(x_list))

    compare(batch, batch_truth)


def compare(ba: Batch, bb: Batch):
    if set(ba.keys) != set(bb.keys):
        raise Exception()
    for k in ba.keys:
        rec_comp(ba[k], bb[k])
        rec_comp(ba._slice_dict[k], bb._slice_dict[k])
        rec_comp(ba._inc_dict[k], bb._inc_dict[k])


def rec_comp(a, b):
    if not type(a) == type(b):
        raise Exception()
    if isinstance(a, dict):
        if not set(a.keys()) == set(b.keys()):
            raise Exception()
        for k in a:
            rec_comp(a[k], b[k])
    if isinstance(a, torch.Tensor):
        if not (a == b).all():
            raise Exception()
