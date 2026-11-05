
import math

from collections import deque

import torch

from torch_geometric.data import Data

from .tree_parser import PropagationTree


# Fixed class mapping across all experiments.
# Matches the class order in Person 1's report.
LABEL_TO_INDEX = {
    "false": 0,
    "non-rumor": 1,
    "true": 2,
    "unverified": 3,
}

NUM_CLASSES = len(LABEL_TO_INDEX)

# Feature order must stay consistent.
FEATURE_NAMES = (
    "is_root",
    "relative_depth",
    "signed_log_delay",
    "log_child_count",
)


def tree_to_graph(
    tree: PropagationTree,
    label: str,
    *,
    bidirectional: bool = True,
) -> Data:
    """
    Convert a parsed propagation tree into a PyG graph.

    Parameters
    ----------
    tree:
        PropagationTree produced by parse_tree().

    label:
        One of the four misinformation classes.

    bidirectional:
        If True, add a reverse edge for every tree edge.

    Returns
    -------
    Data:
        PyTorch Geometric graph with x, edge_index, y.
    """

    # 1. Validate the classification label.
    if label not in LABEL_TO_INDEX:
        raise ValueError(f"Unknown label: {label}")

    num_nodes = len(tree.nodes)
    root = tree.root_index

    if num_nodes == 0:
        raise ValueError("Tree contains no nodes.")

    if not 0 <= root < num_nodes:
        raise ValueError("Invalid root index.")

    # 2. Reconstruct parent-child adjacency.
    children = [[] for _ in range(num_nodes)]

    for parent, child in tree.edges:
        if not (
            0 <= parent < num_nodes
            and 0 <= child < num_nodes
        ):
            raise ValueError("Invalid edge index.")

        children[parent].append(child)

    # 3. Calculate node depths using BFS.
    depths = [-1] * num_nodes
    depths[root] = 0

    queue = deque([root])

    while queue:
        parent = queue.popleft()

        for child in children[parent]:
            if depths[child] != -1:
                raise ValueError(
                    "Tree contains a cycle or multiple paths."
                )

            depths[child] = depths[parent] + 1
            queue.append(child)

    if any(depth == -1 for depth in depths):
        raise ValueError(
            "Tree contains unreachable nodes."
        )

    max_depth = max(1, max(depths))

    # 4. Generate numerical features for each node.
    feature_rows = []

    for index, node in enumerate(tree.nodes):

        is_root = float(index == root)

        relative_depth = depths[index] / max_depth

        # Compress time while preserving its sign.
        delay = node.delay_minutes

        if not math.isfinite(delay):
            raise ValueError("Non-finite time delay.")

        signed_log_delay = math.copysign(
            math.log1p(abs(delay)),
            delay,
        )

        # Number of direct child events.
        log_child_count = math.log1p(
            len(children[index])
        )

        feature_rows.append([
            is_root,
            relative_depth,
            signed_log_delay,
            log_child_count,
        ])

    x = torch.tensor(
        feature_rows,
        dtype=torch.float32,
    )

    # 5. Construct edge_index.
    edge_pairs = list(tree.edges)

    if bidirectional:
        reverse_edges = [
            (child, parent)
            for parent, child in tree.edges
        ]

        edge_pairs.extend(reverse_edges)

    if edge_pairs:
        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long,
        ).t().contiguous()
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    # 6. Encode the graph-level label.
    y = torch.tensor(
        [LABEL_TO_INDEX[label]],
        dtype=torch.long,
    )

    # 7. Construct the PyG Data object.
    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    # Keep the original ID for auditing and tracking.
    graph.source_tweet_id = tree.source_tweet_id

    return graph
