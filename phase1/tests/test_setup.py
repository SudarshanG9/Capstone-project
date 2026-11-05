
import torch
import rumor_graph
from torch_geometric.data import Data


def test_package_import():
    assert rumor_graph is not None


def test_graph_library():
    x = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    edge_index = torch.tensor([
        [0, 1],
        [1, 0]
    ], dtype=torch.long)

    graph = Data(
        x=x,
        edge_index=edge_index
    )

    assert graph.num_nodes == 2
    assert graph.num_edges == 2
