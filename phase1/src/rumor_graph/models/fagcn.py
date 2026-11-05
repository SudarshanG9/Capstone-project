
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import FAConv, global_mean_pool


class FAGCNClassifier(nn.Module):
    """
    FAGCN-style graph classifier using signed FAConv layers.
    """

    def __init__(
        self,
        in_channels=4,
        hidden_channels=32,
        num_classes=4,
        dropout=0.2,
        eps=0.1,
    ):
        super().__init__()

        self.dropout = dropout

        # Project original node features to hidden space.
        self.input_projection = nn.Linear(
            in_channels,
            hidden_channels,
        )

        # Frequency-adaptive graph convolutions.
        self.conv1 = FAConv(
            channels=hidden_channels,
            eps=eps,
            dropout=0.0,
            cached=False,
        )

        self.conv2 = FAConv(
            channels=hidden_channels,
            eps=eps,
            dropout=0.0,
            cached=False,
        )

        # Graph-level classification head.
        self.classifier = nn.Linear(
            hidden_channels,
            num_classes,
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        batch = getattr(data, "batch", None)

        if batch is None:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        # Initial hidden node representation.
        x = self.input_projection(x)
        x = F.relu(x)

        # Keep initial hidden features for FAConv.
        x_0 = x

        # First frequency-adaptive layer.
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv1(x, x_0, edge_index)

        # Second frequency-adaptive layer.
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, x_0, edge_index)

        # Graph-level representation.
        x = global_mean_pool(x, batch)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        logits = self.classifier(x)

        return logits
