"""Unit tests for Rumor GCN and KAN models."""

import pytest
import torch
import torch.nn.functional as F

from rumor_graph.models.gcn import RumorGCN
from rumor_graph.models.kan import KANLinear, RumorKAN
from rumor_graph.train import collate_graphs



from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, FAConv

from rumor_graph.models import (
    GCNClassifier,
    FAGCNClassifier,
)


def test_gcn_forward():
    model = RumorGCN(in_dim=32, hidden_dim=16, num_classes=4)
    x = torch.randn(10, 32)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    out = model(x, edge_index, batch)
    assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"


def test_kan_linear():
    layer = KANLinear(in_features=16, out_features=8, grid_size=3, spline_order=2)
    x = torch.randn(4, 16)
    out = layer(x)
    assert out.shape == (4, 8), f"Expected shape (4, 8), got {out.shape}"


def test_rumor_kan_forward():
    model = RumorKAN(in_dim=32, hidden_dim=16, num_classes=4, grid_size=3, spline_order=2)
    x = torch.randn(8, 32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    out = model(x, edge_index, batch)
    assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"


def test_collate_graphs():
    g1 = {
        "x": torch.randn(4, 16),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "y": torch.tensor([0], dtype=torch.long),
    }
    g2 = {
        "x": torch.randn(3, 16),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "y": torch.tensor([1], dtype=torch.long),
    }

    batched = collate_graphs([g1, g2])
    assert batched["x"].shape == (7, 16)
    assert batched["batch"].shape == (7,)
    assert batched["y"].shape == (2,)






def make_graph(label=0):
    """A small synthetic graph with four node features."""

    return Data(
        x=torch.tensor([
            [1.0, 0.0, 0.0, 1.1],
            [0.0, 0.5, 0.4, 0.7],
            [0.0, 0.5, 0.7, 0.0],
        ], dtype=torch.float32),

        edge_index=torch.tensor([
            [0, 1, 0, 2],
            [1, 0, 2, 0],
        ], dtype=torch.long),

        y=torch.tensor([label], dtype=torch.long),
    )


def make_single_node_graph(label=2):
    """A graph without any parent-child edges."""

    return Data(
        x=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        edge_index=torch.empty(
            (2, 0),
            dtype=torch.long,
        ),
        y=torch.tensor([label], dtype=torch.long),
    )


@pytest.mark.parametrize(
    "model_class",
    [GCNClassifier, FAGCNClassifier],
)
def test_single_graph_forward(model_class):
    torch.manual_seed(42)

    model = model_class(
        in_channels=4,
        hidden_channels=16,
        num_classes=4,
        dropout=0.0,
    )

    model.eval()

    graph = make_graph()
    logits = model(graph)

    assert logits.shape == (1, 4)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "model_class",
    [GCNClassifier, FAGCNClassifier],
)
def test_batch_forward_and_backward(model_class):
    torch.manual_seed(42)

    model = model_class(
        in_channels=4,
        hidden_channels=16,
        num_classes=4,
        dropout=0.0,
    )

    graphs = [
        make_graph(label=0),
        make_single_node_graph(label=2),
    ]

    loader = DataLoader(
        graphs,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    model.train()

    # Forward pass
    logits = model(batch)

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()

    # Classification loss
    loss = F.cross_entropy(
        logits,
        batch.y,
    )

    assert torch.isfinite(loss)

    # Backward pass
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(g is not None for g in gradients)

    assert all(
        torch.isfinite(g).all()
        for g in gradients
    )


@pytest.mark.parametrize(
    "model_class",
    [GCNClassifier, FAGCNClassifier],
)
def test_single_node_graph(model_class):
    model = model_class(
        in_channels=4,
        hidden_channels=16,
        num_classes=4,
        dropout=0.0,
    )

    model.eval()

    logits = model(make_single_node_graph())

    assert logits.shape == (1, 4)
    assert torch.isfinite(logits).all()


def test_correct_convolution_types():
    gcn = GCNClassifier()
    fagcn = FAGCNClassifier()

    assert isinstance(gcn.conv1, GCNConv)
    assert isinstance(gcn.conv2, GCNConv)

    assert isinstance(fagcn.conv1, FAConv)
    assert isinstance(fagcn.conv2, FAConv)


def test_fagcn_attention_range():
    torch.manual_seed(42)

    model = FAGCNClassifier(
        in_channels=4,
        hidden_channels=16,
        dropout=0.0,
    )

    model.eval()

    graph = make_graph()

    x_0 = F.relu(
        model.input_projection(graph.x)
    )

    # Inspect the attention coefficients from FAConv.
    output, (edges, alpha) = model.conv1(
        x_0,
        x_0,
        graph.edge_index,
        return_attention_weights=True,
    )

    assert output.shape == (3, 16)
    assert alpha.numel() == edges.shape[1]
    assert torch.isfinite(alpha).all()

    assert (alpha >= -1.0 - 1e-6).all()
    assert (alpha <= 1.0 + 1e-6).all()
