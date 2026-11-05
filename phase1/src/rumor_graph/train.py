"""Training script for Rumor GCN and KAN models."""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from rumor_graph.models.gcn import RumorGCN
from rumor_graph.models.kan import RumorKAN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RumorDataset(Dataset):
    def __init__(self, graphs: List[Dict]):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_graphs(batch: List[Dict]):
    """Collate individual graph dicts into a batched graph."""
    num_nodes_total = 0
    all_x = []
    all_edges = []
    all_batch = []
    all_y = []

    for i, g in enumerate(batch):
        x = g["x"]
        edge_index = g["edge_index"]
        y = g["y"]
        num_nodes = x.size(0)

        all_x.append(x)
        all_edges.append(edge_index + num_nodes_total)
        all_batch.append(torch.full((num_nodes,), i, dtype=torch.long))
        all_y.append(y)

        num_nodes_total += num_nodes

    batched_x = torch.cat(all_x, dim=0)
    batched_edge_index = torch.cat(all_edges, dim=1)
    batched_batch = torch.cat(all_batch, dim=0)
    batched_y = torch.cat(all_y, dim=0)

    return {
        "x": batched_x,
        "edge_index": batched_edge_index,
        "batch": batched_batch,
        "y": batched_y,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            b = batch["batch"].to(device)
            y = batch["y"].to(device)

            logits = model(x, edge_index, b)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }
    for idx, f1 in enumerate(per_class_f1):
        metrics[f"class_{idx}_f1"] = float(f1)

    return metrics


def train():
    parser = argparse.ArgumentParser(description="Train Rumor Detection Graph Model")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "kan"])
    parser.add_argument("--dataset", type=str, default="twitter15")
    parser.add_argument("--data_file", type=Path, default=Path("processed/twitter15_graphs.pt"))
    parser.add_argument("--runs_dir", type=Path, default=Path("runs"))
    parser.add_argument("--in_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Check processed data file
    if not args.data_file.exists():
        logger.warning(f"Data file {args.data_file} not found! Run rumor_graph.prepare first or synthetic mock will be used.")
        from rumor_graph.prepare import build_graph_data
        graphs = [build_graph_data([str(n) for n in range(6)], [(p, p + 1) for p in range(5)], i % 4, args.in_dim) for i in range(24)]
    else:
        graphs = torch.load(args.data_file, weights_only=False)

    # Train / test split (80/20)
    split_idx = int(len(graphs) * 0.8)
    train_graphs = graphs[:split_idx]
    test_graphs = graphs[split_idx:] if split_idx < len(graphs) else graphs

    train_loader = DataLoader(RumorDataset(train_graphs), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    test_loader = DataLoader(RumorDataset(test_graphs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

    # Model initialization
    if args.model == "gcn":
        model = RumorGCN(in_dim=args.in_dim, hidden_dim=args.hidden_dim, num_classes=args.num_classes).to(device)
    else:
        model = RumorKAN(in_dim=args.in_dim, hidden_dim=args.hidden_dim, num_classes=args.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    out_run_dir = args.runs_dir / args.model
    out_run_dir.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = -1.0
    best_metrics = {}

    logger.info(f"Starting training for {args.model.upper()} ({args.epochs} epochs)...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            b = batch["batch"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logits = model(x, edge_index, b)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate(model, test_loader, device)

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_metrics = metrics
            torch.save(model.state_dict(), out_run_dir / "best_model.pt")

    with open(out_run_dir / "metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    logger.info(f"Training completed. Best metrics saved to {out_run_dir / 'metrics.json'}")


if __name__ == "__main__":
    train()

