
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool


class GCNClassifier(nn.Module):
    """
    Two-layer GCN for graph-level misinformation classification.
    """

    def __init__(
        self,
        in_channels=4,
        hidden_channels=32,
        num_classes=4,
        dropout=0.2,
    ):
        super().__init__()

        self.dropout = dropout

        # First graph convolution:
        # 4 input features -> 32 hidden features
        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        # Second graph convolution:
        # 32 hidden features -> 32 hidden features
        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels,
        )

        # Graph classification layer:
        # 32 graph features -> 4 class scores
        self.classifier = nn.Linear(
            hidden_channels,
            num_classes,
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        # A single Data object may not have a batch vector.
        batch = getattr(data, "batch", None)

        if batch is None:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Convert node features into graph features
        x = global_mean_pool(x, batch)

        # Final graph-level class scores
        logits = self.classifier(x)

        return logits
