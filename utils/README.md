# Data Generation and Preprocessing Pipeline

This document describes the complete workflow for generating synthetic graph datasets, extracting subgraphs via network dismantling, computing features and labels, and preparing train/validation/test splits.

---

## Table of Contents

1. [gen_data.py — Synthetic Network Generation](#1-gen_datapy--synthetic-network-generation)
2. [subnet_gen.py — Subgraph Dataset Generation](#2-subnet_genpy--subgraph-dataset-generation)
3. [process_network.py — Dismantling Sequence Subgraph Generation](#3-process_networkpy--dismantling-sequence-subgraph-generation)
4. [aggregate_remnants.py — Remnant Label and Baseline Aggregation](#4-aggregate_remnantspy--remnant-label-and-baseline-aggregation)

---

## 1. gen_data.py — Synthetic Network Generation

### How to Use

**Using the `--jobs` parameter (recommended):**

1. **Generate 5000 BA networks (range 100–200, IDs starting from 0) and 3000 WS networks (range 100–200, IDs also starting from 0):**
   ```bash
   python gen_data.py --jobs BA:100-200:5000 WS:100-200:3000 --workers 31
   ```

2. **Generate 10000 ER networks (range 200–300), with IDs starting from 5000:**
   ```bash
   python gen_data.py --jobs ER:200-300:10000:5000 --workers 31
   ```

3. **Execute multiple complex, independently-counted tasks in a single command:**
   ```bash
   python gen_data.py --jobs BA:100-200:5000:0 WS:100-200:3000:0 ER:200-300:10000:5000 --workers 31
   ```

**Using simple mode:**

1. **Generate 100 BA networks (range 100–200) and 100 WS networks (range 100–200):**
   ```bash
   python gen_data.py --types BA WS --ranges 100-200 --num_samples 100
   ```
   - This creates `./data_synth/BA-100` and `./data_synth/WS-100` directories.
   - Files will be named `BA-100_0.npz`, `BA-100_1.npz`, …

2. **Generate ER networks for multiple ranges, starting from ID 1000:**
   ```bash
   python gen_data.py --types ER --ranges 300-400 500-600 --num_samples 50 --start_id 1000
   ```
   - This creates `./data_synth/ER-300` and `./data_synth/ER-500` directories.
   - IDs are numbered consecutively starting from 1000.

3. **Generate networks with a non-standard range interval:**
   ```bash
   python gen_data.py --types BA --ranges 100-250 --num_samples 20
   ```
   - This creates `./data_synth/BA-100-250` directory.
   - Files will be named `BA-100-250_0.npz`, …

### Generator Types and Parameter Randomisation

The script implements four mainstream network generation models with randomised parameters to ensure dataset diversity, which is critical for improving model generalisation.

| Model | Full Name | Purpose | Randomised Parameters | Key Characteristics |
|---|---|---|---|---|
| **ER** | Erdős–Rényi | Fully random graphs | Node count `n`; average degree `avg_degree` ∈ \[2, min(8, n/2)\] | Simplest structure; degree distribution approximates a Poisson distribution |
| **BA** | Barabási–Albert | Scale-free networks (preferential attachment / "rich-get-richer") | Node count `n`; edges per new node `m` ∈ \[1, min(8, n/4)\] | Power-law degree distribution; a few high-degree hub nodes; resembles many real-world networks |
| **WS** | Watts–Strogatz | Small-world networks | Node count `n`; neighbour count `k` (random even); rewiring probability `p` ∈ \[0.01, 0.9\] | High clustering coefficient (like regular lattices) combined with short average path length (like random graphs) |
| **LFR** | Lancichinetti–Fortunato–Radicchi | Realistic benchmark graphs with **scale-free properties and community structure** | `tau1`, `tau2` (degree / community-size distribution exponents); `mu` (mixing parameter); `avg_degree`; all heavily randomised | Most advanced generator; produces topologically complex networks with realistic community structure |

### Node Feature Extraction

The script computes a comprehensive set of node-level features for every generated network. These features serve as direct input to downstream models. Under `feature_set='full'`:

| Level | Feature | Description |
|---|---|---|
| **Basic** | `degree` | Node importance / connection count |
| | `clustering` | How tightly a node's neighbours are interconnected |
| | `kcore` | How central a node is within the network's core structure |
| **Extended** | `avg_neighbor_deg` | Whether a node's neighbours are themselves "important" |
| | `pagerank` | Global influence score |
| **Full** | `betweenness` | Ability of a node to act as a bridge in the network |
| | `eigenvector` | Importance weighted by the importance of neighbours (a PageRank variant) |

---

## 2. subnet_gen.py — Subgraph Dataset Generation

> **Location:** `TI-GIN/datasets/subnet_gen.py`

A modular tool for generating large-scale subgraph datasets through network dismantling. See the [subnet_gen.py source documentation](utils/subnet_gen.py) for full API details.

---

## 3. process_network.py — Dismantling Sequence Subgraph Generation

> **Location:** `experiments/5_exploratory_analysis/process_network.py`

### Overview

This tool automates the generation of all subgraphs that arise during the network dismantling process. It operates on node-removal sequences recorded in CSV or XLSX files (produced by various attack algorithms), and also supports generating random removal sequences. Multi-core parallel processing enables fast handling of large-scale datasets.

### Complete Workflow

This tool is a key step in the overall research pipeline:

1. **Prepare initial data** — Provide the original network's `_edges.npz` (edges) and `_features.npy` (features) files.

2. **Compute attack sequences** *(optional)* — Convert networks to `graph-tool` (`.gt`) format, run multiple attack algorithms, and save results (especially the `removals` sequence) as `.csv` or `.xlsx` files.

3. **Generate dismantling subgraphs** — **← This tool performs this step.** It reads the original network files and algorithm result files, then generates all subgraphs along the dismantling trajectory (both `-Remnants` and `-Components`).

4. **Compute subgraph labels** *(subsequent step)* — Convert the generated dismantling subgraphs back to `.gt` format, compute their labels (e.g. robustness metrics), and save as new `.csv` or `.xlsx` files.

5. **Model training and testing** *(subsequent step)* — Use the generated subgraphs and their labels to train and evaluate prediction models (e.g. accuracy, normalised monotonic deviation, etc.).

### Naming Conventions

| Entity | Convention | Example |
|---|---|---|
| **Directory** | Dataset name | `BA-100`, `London` |
| **Original network** | Network name | `BA-100_1`, `REDDIT_1` |
| **Algorithm identifier** | Network–Algorithm | `BA-100_1-DF`, `REDDIT_1-GDMR`, `BA-100_1-R1` |
| **Remnant subgraph** | `NETWORK-ALGO_STEP_edges.npz` — disjoint union of all qualifying connected components after removing `STEP` nodes | `REDDIT_1-GDMR_2_edges.npz` |
| **Component subgraph** | `NETWORK-ALGO_STEP_INDEX_edges.npz` — the `INDEX`-th individual connected component after removing `STEP` nodes | `REDDIT_1-GDMR_2_1_edges.npz` |

### File Structure

**Before running:**

```
/path/to/your/datasets/
└── BA-100/                    ← Dataset root directory (passed as -d)
    ├── BA-100_1_edges.npz
    ├── BA-100_1_features.npy
    ├── BA-100_2_edges.npz
    ├── BA-100_2_features.npy
    ├── ...
    ├── results_part1.csv      ← Algorithm result file 1 (not required for random-only runs)
    └── results_part2.xlsx     ← Algorithm result file 2 (multiple files supported)
```

**After running:**

```
/path/to/your/datasets/
└── BA-100/
    ├── BA-100-Components/     ← Output 1: Individual connected components
    │   ├── BA-100_1-CI1_0_1_edges.npz
    │   ├── BA-100_1-CI1_0_1_features.npy
    │   └── ...
    │
    ├── BA-100-Remnants/       ← Output 2: Disjoint unions of components
    │   ├── BA-100_1-CI1_0_edges.npz
    │   ├── BA-100_1-CI1_0_features.npy
    │   └── ...
    │
    ├── BA-100_1_edges.npz     ← Original files remain unchanged
    ├── ...
    └── results_part1.csv
```

### Command-Line Arguments

| Short | Long | Description | Required | Default |
|:------|:-----|:------------|:---------|:--------|
| `-d` | `--dir` | Path to the dataset root directory | **Yes** | — |
| `-s` | `--size` | Minimum node count for retained connected components | No | `2` |
| `-w` | `--workers` | Number of CPU cores for parallel processing | No | All available |
| `-a` | `--algos` | Algorithms to process (space-separated) | No | All found in result files |
| `-r` | `--random-runs` | Number of random dismantling sequences to generate | No | `0` |

### Examples

1. **Basic usage** — process all algorithms found in result files for the `data/BA-20-100` directory:
   ```bash
   python process_network.py -d data/BA-20-100 -s 4
   ```

2. **Specify worker count** — use 8 cores:
   ```bash
   python process_network.py -d data/BA-20-100 -s 4 -w 8
   ```

3. **Run only specific algorithms** — process only `CI1` and `DCR`:
   ```bash
   python process_network.py -d data/BA-20-100 -s 4 -a CI1 DCR
   ```

4. **Run random dismantling only** — execute 5 random sequences (named `R1` through `R5`):
   ```bash
   python process_network.py -d data/BA-20-100 -s 4 -r 5
   ```

5. **Mixed run** — run `GDM`, `GDMR` and 2 random dismantling sequences:
   ```bash
   python process_network.py -d data/ER-20-100 -a BCR DCR CoreHD GDMR -r 2 -s 4
   ```

---

## 4. aggregate_remnants.py — Remnant Label and Baseline Aggregation

> **Location:** `experiments/5_exploratory_analysis/data/aggregate_remnants.py`

Generates labels and baseline result files for networks in the Remnants directory. **Prerequisite:** the Components directory must already contain processed result files (`results_final`) and label files.

### Step 1 — Consolidate component results

Run `network_data_consolidator.py` on the Components directory first:

```bash
python datasets/network_data_consolidator.py \
    -n /root/autodl-tmp/data_exploratory/power/power-Components/power-Components \
    -r /root/autodl-tmp/data_exploratory/power/power-Components/results \
    -o /root/autodl-tmp/data_exploratory/power/power-Components/results_final
```

### Step 2 — Aggregate into remnants

```bash
python experiments/5_exploratory_analysis/data/aggregate_remnants.py \
    --components-dir /root/autodl-tmp/data_exploratory/power/power-Components \
    --remnants-dir /root/autodl-tmp/data_exploratory/power/power-Remnants
```

```bash
python experiments/5_exploratory_analysis/data/aggregate_remnants.py \
    --components-dir experiments/5_exploratory_analysis/data/synth-20-100/synth-20-100-Components \
    --remnants-dir experiments/5_exploratory_analysis/data/synth-20-100/synth-20-100-Remnants
```

