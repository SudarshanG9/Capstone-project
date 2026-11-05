# Phase 1: Rumor Detection with Graph Models (GCN & KAN)

This repository contains Phase 1 implementation for graph-based rumor detection on Twitter propagation trees from the ACL 2017 benchmark dataset (Twitter15 and Twitter16).

---

## 📁 Directory Structure

```text
phase1/
│
├── data/
│   ├── raw/
│   │   └── rumor_detection_acl2017/
│   │       ├── twitter15/
│   │       │   └── tree/
│   │       └── twitter16/
│   │           └── tree/
│   │
│   └── splits/
│       ├── train_split.csv
│       ├── test_split.csv
│       └── text_baseline_metrics.csv
│
├── src/
│   └── rumor_graph/
│       ├── __init__.py
│       ├── prepare.py
│       ├── train.py
│       │
│       └── models/
│           ├── __init__.py
│           ├── gcn.py
│           └── kan.py
│
├── tests/
│   ├── __init__.py
│   └── test_models.py
│
├── processed/
│
├── runs/
│   ├── gcn/
│   └── kan/
│
├── pyproject.toml
└── README.md
```

---

## 🎯 Task & Dataset Details

The task is **4-class Rumor Stance / Veracity Detection**:
- **0: NR** (Non-Rumor)
- **1: FR** (False Rumor)
- **2: TR** (True Rumor)
- **3: UR** (Unverified Rumor)

### Raw Propagation Trees
Place raw tree files from the ACL 2017 rumor detection release into:
- `data/raw/rumor_detection_acl2017/twitter15/tree/<root_id>.txt`
- `data/raw/rumor_detection_acl2017/twitter16/tree/<root_id>.txt`

Each file contains propagation links:
```text
['parent_id', 'time_delay'] -> ['child_id', 'time_delay']
```

---

## 🚀 Setup & Installation

From the `phase1/` directory:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install package in editable mode with dependencies
pip install -e ".[dev]"
```

---

## 🛠️ Usage

### 1. Data Preparation
Convert raw propagation tree files into preprocessed graph tensors:

```bash
# For Twitter15
python -m rumor_graph.prepare --dataset twitter15

# For Twitter16
python -m rumor_graph.prepare --dataset twitter16
```
Processed graph datasets are saved to `processed/<dataset>_graphs.pt`.

### 2. Training Models

#### Train Graph Convolutional Network (GCN):
```bash
python -m rumor_graph.train --model gcn --dataset twitter15 --epochs 20 --batch_size 16
```

#### Train Kolmogorov-Arnold Network (KAN):
```bash
python -m rumor_graph.train --model kan --dataset twitter15 --epochs 20 --batch_size 16
```

Checkpoints and evaluation metrics (`metrics.json`, `best_model.pt`) will be saved in:
- `runs/gcn/`
- `runs/kan/`

---

## 🧪 Testing

Run unit tests with `pytest`:

```bash
pytest tests/
```

---

## 📊 Baseline Comparison

Text baseline results are tracked in `data/splits/text_baseline_metrics.csv` to benchmark against graph propagation models (GCN, KAN).

