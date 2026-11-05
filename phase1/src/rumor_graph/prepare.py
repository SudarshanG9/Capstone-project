"""Data preparation and graph construction for Twitter rumor datasets."""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP = {"NR": 0, "non-rumor": 0, "FR": 1, "false": 1, "TR": 2, "true": 2, "UR": 3, "unverified": 3}


def parse_tree_file(file_path: Path) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Parse an ACL 2017 propagation tree file.
    
    Lines typically follow format:
    ['parent_id', 'parent_time'] -> ['child_id', 'child_time']
    or: parent_id -> child_id
    """
    nodes: Dict[str, int] = {}
    edges: List[Tuple[int, int]] = []

    def get_node_id(node_str: str) -> int:
        clean_str = re.sub(r"[\[\]'\"\s]", "", node_str).split(",")[0]
        if clean_str not in nodes:
            nodes[clean_str] = len(nodes)
        return nodes[clean_str]

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            parts = line.split("->")
            if len(parts) == 2:
                src_str, dst_str = parts[0].strip(), parts[1].strip()
                if "None" in src_str or "ROOT" in src_str:
                    get_node_id(dst_str)
                    continue
                src = get_node_id(src_str)
                dst = get_node_id(dst_str)
                edges.append((src, dst))

    node_list = list(nodes.keys())
    return node_list, edges


def build_graph_data(
    node_list: List[str],
    edges: List[Tuple[int, int]],
    label: int,
    feature_dim: int = 128,
) -> dict:
    """Build graph tensors (features, edge_index, label)."""
    num_nodes = max(len(node_list), 1)

    # Initialize node features (can be replaced by pre-extracted text embeddings)
    x = torch.randn(num_nodes, feature_dim, dtype=torch.float)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Self-loop for disconnected root
        edge_index = torch.zeros((2, 1), dtype=torch.long)

    y = torch.tensor([label], dtype=torch.long)

    return {
        "x": x,
        "edge_index": edge_index,
        "y": y,
        "num_nodes": num_nodes,
    }


def prepare_dataset(
    dataset_name: str,
    raw_dir: Path,
    splits_dir: Path,
    out_dir: Path,
    feature_dim: int = 128,
    generate_synthetic_if_empty: bool = True,
):
    """Preprocess propagation trees into processed graph tensors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_dir = raw_dir / dataset_name / "tree"
    tree_files = list(tree_dir.glob("*.txt")) if tree_dir.exists() else []

    split_file = splits_dir / "train_split.csv"
    labels_by_id = {}
    if split_file.exists():
        df_split = pd.read_csv(split_file)
        for _, row in df_split.iterrows():
            lbl = row.get("label_idx", LABEL_MAP.get(str(row.get("label")), 0))
            labels_by_id[str(row["root_id"])] = int(lbl)

    processed_data = []

    if not tree_files and generate_synthetic_if_empty:
        logger.warning(f"No tree files found in {tree_dir}. Generating mock graph samples for verification.")
        for i in range(16):
            root_id = f"mock_{dataset_name}_{i}"
            num_nodes = 5 + (i % 7)
            edges = [(p, c) for p in range(num_nodes - 1) for c in [p + 1]]
            label = i % 4
            graph = build_graph_data([str(n) for n in range(num_nodes)], edges, label, feature_dim)
            graph["root_id"] = root_id
            processed_data.append(graph)
    else:
        logger.info(f"Processing {len(tree_files)} propagation trees from {tree_dir}...")
        for tree_path in tree_files:
            root_id = tree_path.stem
            nodes, edges = parse_tree_file(tree_path)
            label = labels_by_id.get(root_id, 0)
            graph = build_graph_data(nodes, edges, label, feature_dim)
            graph["root_id"] = root_id
            processed_data.append(graph)

    out_file = out_dir / f"{dataset_name}_graphs.pt"
    torch.save(processed_data, out_file)
    logger.info(f"Saved {len(processed_data)} processed graphs to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Twitter Rumor Graph Data")
    parser.add_argument("--dataset", type=str, default="twitter15", choices=["twitter15", "twitter16"])
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw/rumor_detection_acl2017"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--out_dir", type=Path, default=Path("processed"))
    parser.add_argument("--feature_dim", type=int, default=128)
    args = parser.parse_args()

    prepare_dataset(
        dataset_name=args.dataset,
        raw_dir=args.raw_dir,
        splits_dir=args.splits_dir,
        out_dir=args.out_dir,
        feature_dim=args.feature_dim,
    )


if __name__ == "__main__":
    main()

