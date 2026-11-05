
import pytest
import torch

from torch_geometric.loader import DataLoader

from rumor_graph.tree_parser import parse_tree
from rumor_graph.graph_builder import (
    FEATURE_NAMES,
    LABEL_TO_INDEX,
    tree_to_graph,
)


@pytest.fixture
def sample_tree(tmp_path):
    """Create and parse a small synthetic tree."""

    lines = [
        "['ROOT','ROOT','0.0']->['u0','100','0.0']",
        "['u0','100','0.0']->['u1','101','0.5']",
        "['u0','100','0.0']->['u2','102','1.0']",
        "['u1','101','0.5']->['u3','103','2.0']",
    ]

    path = tmp_path / "100.txt"

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return parse_tree(path)


def test_graph_shapes(sample_tree):
    graph = tree_to_graph(sample_tree, "false")

    assert graph.num_nodes == 4
    assert graph.x.shape == (4, 4)
    assert graph.edge_index.shape == (2, 6)
    assert graph.y.shape == (1,)

    assert graph.x.dtype == torch.float32
    assert graph.edge_index.dtype == torch.long
    assert graph.y.dtype == torch.long


def test_feature_names():
    assert FEATURE_NAMES == (
        "is_root",
        "relative_depth",
        "signed_log_delay",
        "log_child_count",
    )


def test_root_and_depth_features(sample_tree):
    graph = tree_to_graph(sample_tree, "false")

    # Root indicator
    assert graph.x[:, 0].tolist() == [
        1.0, 0.0, 0.0, 0.0
    ]

    # Relative depth values
    expected = torch.tensor([
        0.0, 0.5, 0.5, 1.0
    ])

    assert torch.allclose(
        graph.x[:, 1],
        expected,
    )


def test_bidirectional_edges(sample_tree):
    graph = tree_to_graph(
        sample_tree,
        "false",
        bidirectional=True,
    )

    edges = set(
        map(tuple, graph.edge_index.t().tolist())
    )

    expected = {
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 0),
        (2, 0),
        (3, 1),
    }

    assert edges == expected


def test_forward_only_edges(sample_tree):
    graph = tree_to_graph(
        sample_tree,
        "false",
        bidirectional=False,
    )

    edges = set(
        map(tuple, graph.edge_index.t().tolist())
    )

    assert edges == {
        (0, 1),
        (0, 2),
        (1, 3),
    }


@pytest.mark.parametrize(
    "label,index",
    [
        ("false", 0),
        ("non-rumor", 1),
        ("true", 2),
        ("unverified", 3),
    ],
)
def test_label_encoding(sample_tree, label, index):
    graph = tree_to_graph(sample_tree, label)

    assert graph.y.item() == index
    assert LABEL_TO_INDEX[label] == index


def test_invalid_label(sample_tree):
    with pytest.raises(ValueError, match="Unknown label"):
        tree_to_graph(sample_tree, "unknown")


def test_source_id_preserved(sample_tree):
    graph = tree_to_graph(sample_tree, "false")

    assert graph.source_tweet_id == "100"


def test_single_node_graph(tmp_path):
    path = tmp_path / "200.txt"

    path.write_text(
        "['ROOT','ROOT','0.0']->['u0','200','0.0']",
        encoding="utf-8",
    )

    tree = parse_tree(path)
    graph = tree_to_graph(tree, "true")

    assert graph.x.shape == (1, 4)
    assert graph.edge_index.shape == (2, 0)
    assert graph.y.item() == 2


def test_negative_delay_is_finite(tmp_path):
    path = tmp_path / "300.txt"

    path.write_text(
        "\n".join([
            "['ROOT','ROOT','0.0']->['u0','300','0.0']",
            "['u0','300','0.0']->['u1','301','-0.5']",
        ]),
        encoding="utf-8",
    )

    tree = parse_tree(path)
    graph = tree_to_graph(tree, "false")

    assert torch.isfinite(graph.x).all()
    assert graph.x[1, 2].item() < 0


def test_graph_batching(sample_tree):
    graph_a = tree_to_graph(sample_tree, "false")
    graph_b = tree_to_graph(sample_tree, "true")

    loader = DataLoader(
        [graph_a, graph_b],
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch.num_graphs == 2
    assert batch.x.shape == (8, 4)
    assert batch.y.tolist() == [0, 2]
    assert batch.batch.shape == (8,)
