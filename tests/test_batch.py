import torch
from torch_geometric.data import Batch, Data

from caloutils.utils.batch import (
    add_graph_attr,
    add_node_attr,
    from_batch_list,
    init_batch,
    set_edge_attr,
    set_edges,
)

device = torch.device("cpu")


def test_batch_add_node_attr():
    batch_size = 3
    node_range = (1, 10)
    nodes_v = torch.randint(*node_range, (batch_size,)).to(device)
    x_list = [torch.rand(n, 3).to(device) for n in nodes_v]

    batch_list = [Data(x=x) for x in x_list]
    batch_truth = Batch.from_data_list(batch_list)

    batch_idx = batch_truth.batch

    batch = init_batch(batch_idx)

    add_node_attr(batch, "x", torch.vstack(x_list))

    compare(batch_truth, batch)


def test_batch_set_edges():
    batch_size = 4
    node_range = (2, 5)
    nodes_v = torch.randint(*node_range, (batch_size,)).to(device)
    edges_per_graph = torch.cat(
        [torch.randint(1, num_nodes, size=(1,)).to(device) for num_nodes in nodes_v]
    ).to(device)
    x_list = [torch.rand(num_nodes, 3).to(device) for num_nodes in nodes_v]
    edges_list = [
        torch.vstack(
            [
                torch.randint(0, num_nodes, size=(num_edges,)),
                torch.randint(0, num_nodes, size=(num_edges,)),
            ]
        ).to(device)
        for num_nodes, num_edges in zip(nodes_v, edges_per_graph)
    ]

    batch_list = [
        Data(x=x, edge_index=edges) for x, edges in zip(x_list, edges_list)
    ]
    batch_truth = Batch.from_data_list(batch_list)

    batch_idx = batch_truth.batch
    batch = init_batch(batch_idx)
    add_node_attr(batch, "x", torch.vstack(x_list))

    batchidx_per_edge = torch.cat(
        [
            torch.ones(num_edges).long().to(device) * igraph
            for igraph, num_edges in enumerate(edges_per_graph)
        ]
    )
    set_edges(batch, torch.hstack(edges_list), batchidx_per_edge)
    compare(batch_truth, batch)


def test_batch_set_edge_attr():
    batch_size = 4
    node_range = (2, 5)
    nodes_v = torch.randint(*node_range, (batch_size,)).to(device)
    edges_per_graph = torch.cat(
        [torch.randint(1, num_nodes, size=(1,)).to(device) for num_nodes in nodes_v]
    )
    x_list = [torch.rand(num_nodes, 3).to(device) for num_nodes in nodes_v]
    edges_list = [
        torch.vstack(
            [
                torch.randint(0, num_nodes, size=(num_edges,)),
                torch.randint(0, num_nodes, size=(num_edges,)),
            ]
        ).to(device)
        for num_nodes, num_edges in zip(nodes_v, edges_per_graph)
    ]
    edge_attr_list = [
        torch.rand(num_edges).to(device) for num_edges in edges_per_graph
    ]

    batch_list = [
        Data(x=x, edge_index=edges, edge_attr=ea)
        for x, edges, ea in zip(x_list, edges_list, edge_attr_list)
    ]
    batch_truth = Batch.from_data_list(batch_list)

    batch_idx = batch_truth.batch
    batch = init_batch(batch_idx)
    add_node_attr(batch, "x", torch.vstack(x_list))

    batchidx_per_edge = torch.cat(
        [
            torch.ones(num_edges).to(device).long() * igraph
            for igraph, num_edges in enumerate(edges_per_graph)
        ]
    )
    set_edges(batch, torch.hstack(edges_list), batchidx_per_edge)
    set_edge_attr(batch, torch.hstack(edge_attr_list), batchidx_per_edge)
    compare(batch_truth, batch)


def test_batch_add_graph_attr():
    batch_size = 3
    node_range = (1, 10)
    nodes_v = torch.randint(*node_range, (batch_size,)).to(device)
    x_list = [torch.rand(n, 3).to(device) for n in nodes_v]
    graph_attr_list = torch.rand(batch_size).to(device)

    batch_list = [Data(x=x, ga=ga) for x, ga in zip(x_list, graph_attr_list)]
    batch_truth = Batch.from_data_list(batch_list)

    batch_idx = batch_truth.batch

    batch = init_batch(batch_idx)

    add_node_attr(batch, "x", torch.vstack(x_list))
    add_graph_attr(batch, "ga", graph_attr_list)
    compare(batch_truth, batch)


def test_from_batch_list():
    batch_size = 9
    node_range = (2, 5)
    nodes_v = torch.randint(*node_range, (batch_size,)).to(device)
    edges_per_graph = torch.cat(
        [torch.randint(1, num_nodes, size=(1,)).to(device) for num_nodes in nodes_v]
    )
    x_list = [torch.rand(num_nodes, 3).to(device) for num_nodes in nodes_v]
    edges_list = [
        torch.vstack(
            [
                torch.randint(0, num_nodes, size=(num_edges,)),
                torch.randint(0, num_nodes, size=(num_edges,)),
            ]
        ).to(device)
        for num_nodes, num_edges in zip(nodes_v, edges_per_graph)
    ]
    edge_attr_list = [
        torch.rand(num_edges).to(device) for num_edges in edges_per_graph
    ]
    graph_attr_list = torch.rand(batch_size).to(device)

    batch_list = [
        Data(x=x, edge_index=edges, edge_attr=ea, ga=ga)
        for x, edges, ea, ga in zip(
            x_list, edges_list, edge_attr_list, graph_attr_list
        )
    ]
    batch_truth = Batch.from_data_list(batch_list)
    batch = from_batch_list(
        Batch.from_data_list(batch_list[:3]), Batch.from_data_list(batch_list[3:])
    )

    compare(batch_truth, batch)


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


test_batch_set_edges()
