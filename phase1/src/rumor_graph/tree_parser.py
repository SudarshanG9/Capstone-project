import ast
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TreeNode:
    user_id: str
    tweet_id: str
    delay_minutes: float


@dataclass(frozen=True)
class PropagationTree:
    source_tweet_id: str
    nodes: tuple[TreeNode, ...]
    edges: tuple[tuple[int, int], ...]
    root_index: int = 0




def parse_node(text: str) -> TreeNode:
    """Read ['user_id', 'tweet_id', 'delay'] safely."""

    try:
        parts = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid node: {text}") from exc

    if not isinstance(parts, (tuple, list)) or len(parts) != 3:
        raise ValueError("A node must have three fields")

    user_id, tweet_id, delay = parts

    if not isinstance(user_id, (str, int)) or not isinstance(
        tweet_id, (str, int)
    ):
        raise ValueError("Node IDs must be strings or integers")

    user_id = str(user_id).strip()
    tweet_id = str(tweet_id).strip()

    if not user_id or not tweet_id:
        raise ValueError("Node IDs cannot be empty")

    try:
        delay_minutes = float(delay)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid delay") from exc

    if not math.isfinite(delay_minutes):
        raise ValueError("Delay must be finite")

    return TreeNode(
        user_id=user_id,
        tweet_id=tweet_id,
        delay_minutes=delay_minutes,
    )



def parse_tree(path: str | Path) -> PropagationTree:
    """Read one Twitter15/16 propagation-tree file."""

    path = Path(path)
    pairs = []

    # 1. Read parent -> child relationships
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.strip()

        if not line:
            continue

        if line.count("->") != 1:
            raise ValueError(
                f"Line {line_no}: expected one '->'"
            )

        left, right = line.split("->", 1)

        try:
            pairs.append((
                parse_node(left),
                parse_node(right),
            ))
        except ValueError as exc:
            raise ValueError(
                f"Line {line_no}: {exc}"
            ) from exc

    # 2. Identify the source tweet
    def is_root_marker(node: TreeNode) -> bool:
        return (
            node.user_id == "ROOT"
            and node.tweet_id == "ROOT"
        )

    roots = [
        child
        for parent, child in pairs
        if is_root_marker(parent)
    ]

    if len(roots) != 1:
        raise ValueError(
            f"Expected exactly one ROOT line; found {len(roots)}"
        )

    source = roots[0]

    if source.tweet_id != path.stem:
        raise ValueError(
            "Source tweet ID does not match filename"
        )

    # 3. Assign unique numerical indices to nodes
    nodes = [source]
    indices = {
        (source.user_id, source.tweet_id): 0
    }

    edges = []
    edge_set = set()
    parents_of = {}

    def add_node(node: TreeNode) -> int:
        key = (node.user_id, node.tweet_id)

        if key in indices:
            idx = indices[key]

            if nodes[idx].delay_minutes != node.delay_minutes:
                raise ValueError(
                    f"Conflicting delay for node {key}"
                )

            return idx

        idx = len(nodes)
        nodes.append(node)
        indices[key] = idx

        return idx

    # 4. Build directed edges
    for parent, child in pairs:

        if is_root_marker(parent):
            continue

        if "ROOT" in (
            parent.user_id,
            parent.tweet_id,
            child.user_id,
            child.tweet_id,
        ):
            raise ValueError(
                "ROOT marker used as a real node"
            )

        parent_id = add_node(parent)
        child_id = add_node(child)

        if parent_id == child_id:
            raise ValueError("Self-loop detected")

        if (
            child_id in parents_of
            and parents_of[child_id] != parent_id
        ):
            raise ValueError(
                "Node has multiple parents"
            )

        parents_of[child_id] = parent_id

        edge = (parent_id, child_id)

        if edge not in edge_set:
            edges.append(edge)
            edge_set.add(edge)

    # 5. Validate tree structure
    if 0 in parents_of:
        raise ValueError(
            "Source tweet cannot have a parent"
        )

    if len(edges) != len(nodes) - 1:
        raise ValueError(
            "Invalid number of edges for a tree"
        )

    # 6. Check that every node is reachable
    children = {
        i: [] for i in range(len(nodes))
    }

    for a, b in edges:
        children[a].append(b)

    visited = set()
    pending = [0]

    while pending:
        current = pending.pop()

        if current in visited:
            continue

        visited.add(current)
        pending.extend(children[current])

    if len(visited) != len(nodes):
        raise ValueError(
            "Disconnected nodes or cycle detected"
        )

    # 7. Return the parsed tree
    return PropagationTree(
        source_tweet_id=path.stem,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )

