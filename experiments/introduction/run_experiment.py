#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/experiments/introduction/run_experiment.py

Pipeline for the introduction experiment (network dismantling).

Subcommands
-----------
generate
    Generate one synthetic network and save:
    data/<name>/<name>_edges.npz
attack
    Run DC / BC / Rand / Opt attacks for one absolute tau, export
    per-step remnants/components, and compute brute-force component labels.
metrics
    Compute per-step metrics for remnant graphs:
    LCC, natural_connectivity, R(rand), R(DCR), collapse_distance.
all
    Run `attack` then `metrics` for the same tau.

Typical workflow (run from project root)
----------------------------------------
# 1) Generate networks
python experiments/introduction/run_experiment.py generate --name net1 --type WS --n 20 --k 6 --p 0.1
python experiments/introduction/run_experiment.py generate --name net2 --type ER --n 20 --avg_degree 4

# 2) Attack at two taus (for N=20: 0.2N=4, 0.5N=10)
python experiments/introduction/run_experiment.py attack --tau 10 --nets net1 net2 --workers 16 --force
python experiments/introduction/run_experiment.py attack --tau 4  --nets net1 net2 --workers 16 --force

# 3) Compute metrics
python experiments/introduction/run_experiment.py metrics --nets net1 net2 --tau 4 10 --output metrics.csv --workers 16 --force

# 4) Optional one-shot run (single tau)
python experiments/introduction/run_experiment.py all --tau 10 --nets net1 net2 --output metrics_tau10.csv --workers 16 --force
"""

import sys
import argparse
import json
import random
import time
import warnings
import multiprocessing as mp
from pathlib import Path
from itertools import combinations
from math import comb
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx
import pandas as pd
from scipy import linalg
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# Section 0. Imports and Path Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

# Add project root and utils dir for optional import
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

try:
    from gen_data import generate_network
except ImportError:
    print("[WARN] Could not import generate_network from utils/gen_data.py. Using built-in fallback.")
    generate_network = None

ATTACK_ALGOS = ["DC", "BC", "Rand", "Opt"]


# =============================================================================
# Section 1. Shared Utilities
# =============================================================================

def load_graph_from_npz(npz_path):
    """Load an undirected graph from NPZ edge list."""
    data = np.load(npz_path)
    edges = data["edges"]
    G = nx.Graph()
    if edges.size > 0:
        G.add_edges_from(edges.tolist())
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def save_graph_npz(G, path):
    """
    Save graph edge list to NPZ after relabeling nodes to contiguous 0..n-1.
    If path has no extension, numpy will append '.npz'.
    """
    if G.number_of_nodes() == 0:
        np.savez_compressed(path, edges=np.array([], dtype=int).reshape(0, 2))
        return

    nodes = sorted(list(G.nodes()))
    mapping = {old: new for new, old in enumerate(nodes)}
    edges = np.array([(mapping[u], mapping[v]) for u, v in G.edges()], dtype=int)
    if edges.size == 0:
        edges = edges.reshape(0, 2)
    np.savez_compressed(path, edges=edges)


def get_tau_values_for_net(net_name):
    """
    Scan data/<net_name>/ and detect all available tau folders.
    Example: net1-5/, net1-10/ -> [5, 10]
    """
    net_dir = DATA_DIR / net_name
    if not net_dir.exists():
        return []

    taus = []
    prefix = f"{net_name}-"
    for d in sorted(net_dir.iterdir()):
        if d.is_dir() and d.name.startswith(prefix):
            suffix = d.name[len(prefix):]
            try:
                tau_val = int(suffix)
                taus.append(tau_val)
            except ValueError:
                continue
    return sorted(taus)


def get_all_net_names():
    """Scan data/ and return all network names with existing edge files."""
    if not DATA_DIR.exists():
        return []

    names = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            edge_file = d / f"{d.name}_edges.npz"
            if edge_file.exists():
                names.append(d.name)
    return names


def _generate_network_builtin(network_type, params):
    """Fallback synthetic graph generator."""
    n = params.get("n", 100)

    if network_type == "ER":
        p = params["avg_degree"] / (n - 1)
        return nx.erdos_renyi_graph(n, p)
    elif network_type == "BA":
        m = params.get("m", 3)
        return nx.barabasi_albert_graph(n, m)
    elif network_type == "WS":
        k = params.get("k", 4)
        p = params.get("p", 0.1)
        return nx.watts_strogatz_graph(n, k, p)
    else:
        raise ValueError(f"Unsupported network type: {network_type}")


# =============================================================================
# Section 2. Phase 1 — Network Generation
# =============================================================================

def phase_generate(args):
    """Generate one network and save it under data/<name>/."""
    name = args.name
    net_dir = DATA_DIR / name
    edge_file = net_dir / f"{name}_edges.npz"

    if edge_file.exists() and not args.force:
        print(f"[SKIP] {name} already exists: {edge_file}")
        return

    # Build generation parameters
    params = {"n": args.n}
    net_type = args.type.upper()

    if net_type == "BA":
        params["m"] = args.m
    elif net_type == "ER":
        params["avg_degree"] = args.avg_degree
    elif net_type == "WS":
        params["k"] = args.k
        params["p"] = args.p
    elif net_type == "LFR":
        params["avg_degree"] = args.avg_degree
        params["tau1"] = getattr(args, "tau1", 2.5)
        params["tau2"] = getattr(args, "tau2", 1.5)
        params["mu"] = getattr(args, "mu", 0.3)
        params["max_degree"] = int(args.n * 0.5)
        params["min_community"] = max(10, int(args.avg_degree))
        params["max_community"] = int(args.n * 0.3)

    print(f"[GEN] Generating network {name}: type={net_type}, params={params}")

    gen_func = generate_network if generate_network else _generate_network_builtin
    max_retries = 20
    G = None

    for attempt in range(max_retries):
        try:
            G_raw = gen_func(net_type, params)
            if G_raw is None or G_raw.number_of_nodes() < 3:
                continue

            # Remove self loops
            G_raw.remove_edges_from(nx.selfloop_edges(G_raw))

            # Ensure connected graph
            if not nx.is_connected(G_raw):
                comps = list(nx.connected_components(G_raw))
                main_nodes = max(comps, key=len)
                G_conn = G_raw.subgraph(main_nodes).copy()

                for comp_nodes in comps:
                    if comp_nodes != main_nodes:
                        n1 = random.choice(list(G_conn.nodes()))
                        n2 = random.choice(list(comp_nodes))
                        G_conn.add_node(n2)
                        G_conn.add_edge(n1, n2)

                        for other in comp_nodes:
                            if other != n2:
                                G_conn.add_node(other)
                                for nbr in G_raw.neighbors(other):
                                    if nbr in G_conn:
                                        G_conn.add_edge(other, nbr)

                G_raw = G_conn

            # Relabel to 0..n-1
            G = nx.convert_node_labels_to_integers(G_raw, first_label=0)

            # Remove isolated nodes
            isolates = list(nx.isolates(G))
            if isolates:
                G.remove_nodes_from(isolates)
                G = nx.convert_node_labels_to_integers(G, first_label=0)

            # Final cleanup/validation
            G.remove_edges_from(nx.selfloop_edges(G))
            nodes = sorted(G.nodes())
            if nodes == list(range(G.number_of_nodes())) and nx.is_connected(G):
                break
            else:
                G = None

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] Failed to generate {name}: {e}")
                return
            continue

    if G is None:
        print(f"[ERROR] Failed to generate {name}: constraints not satisfied after retries.")
        return

    net_dir.mkdir(parents=True, exist_ok=True)
    save_graph_npz(G, str(edge_file))

    avg_deg = round(2 * G.number_of_edges() / G.number_of_nodes(), 4) if G.number_of_nodes() > 0 else 0.0
    meta = {
        "name": name,
        "type": net_type,
        "params": params,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": avg_deg,
    }

    with open(net_dir / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(
        f"[DONE] {name}: N={G.number_of_nodes()}, E={G.number_of_edges()}, "
        f"avg_deg={meta['avg_degree']}, saved to {edge_file}"
    )


# =============================================================================
# Section 3. Attack Strategies and Solvers
# =============================================================================

def attack_dc_static(G, tau):
    """Static degree-centrality attack: one-shot ranking, remove until LCC < tau."""
    deg = dict(G.degree())
    order = sorted(deg.keys(), key=lambda x: (-deg[x], x))
    seq = []
    G_sim = G.copy()

    for node in order:
        if G_sim.number_of_nodes() == 0:
            break
        if not G_sim.has_node(node):
            continue
        G_sim.remove_node(node)
        seq.append(node)
        if G_sim.number_of_nodes() == 0:
            break
        lcc_size = len(max(nx.connected_components(G_sim), key=len))
        if lcc_size < tau:
            break
    return seq


def attack_bc_static(G, tau):
    """Static betweenness-centrality attack: one-shot ranking, remove until LCC < tau."""
    bc = nx.betweenness_centrality(G)
    order = sorted(bc.keys(), key=lambda x: (-bc[x], x))
    seq = []
    G_sim = G.copy()

    for node in order:
        if G_sim.number_of_nodes() == 0:
            break
        if not G_sim.has_node(node):
            continue
        G_sim.remove_node(node)
        seq.append(node)
        if G_sim.number_of_nodes() == 0:
            break
        lcc_size = len(max(nx.connected_components(G_sim), key=len))
        if lcc_size < tau:
            break
    return seq


def attack_random(G, tau):
    """Random attack: random permutation of nodes, remove until LCC < tau."""
    nodes = list(G.nodes())
    random.shuffle(nodes)
    seq = []
    G_sim = G.copy()

    for node in nodes:
        if G_sim.number_of_nodes() == 0:
            break
        G_sim.remove_node(node)
        seq.append(node)
        if G_sim.number_of_nodes() == 0:
            break
        lcc_size = len(max(nx.connected_components(G_sim), key=len))
        if lcc_size < tau:
            break
    return seq


def _is_disrupted(G, tau):
    """Return True if graph is disrupted under criterion LCC < tau."""
    if G.number_of_nodes() < tau:
        return True
    comps = list(nx.connected_components(G))
    if not comps:
        return True
    return len(max(comps, key=len)) < tau


def _heuristic_removal(G, tau, mode="dc"):
    """
    Heuristic dismantling sequence for upper-bound estimation.

    mode:
      - 'dc'  static degree
      - 'bc'  static betweenness
      - 'dcr' dynamic degree (recompute each step)
      - 'bcr' dynamic betweenness (recompute each step)
    """
    G_temp = G.copy()
    removed = []

    if mode == "dc":
        deg = dict(G.degree())
        order = sorted(deg.keys(), key=lambda x: (-deg[x], x))
        for node in order:
            if _is_disrupted(G_temp, tau):
                break
            if G_temp.has_node(node):
                G_temp.remove_node(node)
                removed.append(node)

    elif mode == "bc":
        bc = nx.betweenness_centrality(G)
        order = sorted(bc.keys(), key=lambda x: (-bc[x], x))
        for node in order:
            if _is_disrupted(G_temp, tau):
                break
            if G_temp.has_node(node):
                G_temp.remove_node(node)
                removed.append(node)

    elif mode == "dcr":
        while not _is_disrupted(G_temp, tau):
            if G_temp.number_of_nodes() == 0:
                break
            node = max(G_temp.nodes(), key=lambda x: (G_temp.degree(x), x))
            G_temp.remove_node(node)
            removed.append(node)

    elif mode == "bcr":
        while not _is_disrupted(G_temp, tau):
            if G_temp.number_of_nodes() == 0:
                break
            bc = nx.betweenness_centrality(G_temp)
            node = max(bc, key=lambda x: (bc[x], x))
            G_temp.remove_node(node)
            removed.append(node)

    return removed


def _best_heuristic_upper_bound(G, tau):
    """
    Evaluate DC/BC/DCR/BCR and return best upper bound:
    (min_count, removed_nodes, mode_name).
    """
    best_count = G.number_of_nodes()
    best_nodes = []
    best_mode = None

    for mode in ["dc", "bc", "dcr", "bcr"]:
        try:
            removed = _heuristic_removal(G, tau, mode)
            if len(removed) < best_count:
                best_count = len(removed)
                best_nodes = removed
                best_mode = mode
        except Exception:
            continue

    return best_count, best_nodes, best_mode


def brute_force_min_removal(G, tau):
    """
    Brute-force minimum node-removal set such that LCC < tau.

    Steps
    -----
    1) Compute heuristic upper bound with DC/BC/DCR/BCR.
    2) Exhaustively search from (upper_bound - 1) downwards.

    Returns
    -------
    (best_count, best_nodes)
    """
    if _is_disrupted(G, tau):
        return 0, []

    n = G.number_of_nodes()

    # Step 1: upper bound by heuristics
    best_count, best_nodes, best_mode = _best_heuristic_upper_bound(G, tau)
    best_count = max(1, min(best_count, n - 1))

    total_search = sum(comb(n, k) for k in range(best_count - 1, 0, -1))
    print(
        f"       Heuristic upper bound: {best_count} (from {best_mode}), "
        f"exhaustive search k={best_count - 1}→1, max combinations≈{total_search:,.0f}"
    )

    # Step 2: exhaustive search from k=best_count-1 down to 1
    nodes_by_degree = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)

    for k in range(best_count - 1, 0, -1):
        found = False
        for nodes_to_remove in combinations(nodes_by_degree, k):
            G_temp = G.copy()
            G_temp.remove_nodes_from(nodes_to_remove)
            if _is_disrupted(G_temp, tau):
                best_count = k
                best_nodes = list(nodes_to_remove)
                found = True
                break

        if not found:
            # No disruptive set at size k -> stop descending
            break

    return best_count, best_nodes


def attack_optimal(G, tau):
    """
    Optimal attack:
    brute-force minimum removal set, then order nodes by degree (descending).
    """
    if _is_disrupted(G, tau):
        return []

    k, opt_nodes = brute_force_min_removal(G, tau)
    if k == 0:
        return []

    deg = dict(G.degree())
    opt_nodes_sorted = sorted(opt_nodes, key=lambda x: (-deg.get(x, 0), x))
    return opt_nodes_sorted


# =============================================================================
# Section 4. Phase 2 — Attack Execution, Sequence Export, and Labeling
# =============================================================================

def generate_sequence_files(G_original, removal_seq, net_name, algo_name, tau, comp_dir, rem_dir):
    """
    Generate remnant/component graph files from one removal sequence.

    Step naming:
      remnant : <net>-<algo>_<step>_edges.npz
      component: <net>-<algo>_<step>_<idx>_edges.npz
    """
    G_sim = G_original.copy()

    # Step 0: intact graph
    save_graph_npz(G_sim, str(rem_dir / f"{net_name}-{algo_name}_0_edges"))
    comps_0 = [G_sim.subgraph(c).copy() for c in nx.connected_components(G_sim)]
    valid_0 = [c for c in comps_0 if c.number_of_nodes() >= tau]
    for idx, comp in enumerate(valid_0, 1):
        save_graph_npz(comp, str(comp_dir / f"{net_name}-{algo_name}_0_{idx}_edges"))

    # Remove nodes step by step
    for step, node in enumerate(removal_seq, 1):
        if G_sim.has_node(node):
            G_sim.remove_node(node)

        if G_sim.number_of_nodes() == 0:
            save_graph_npz(G_sim, str(rem_dir / f"{net_name}-{algo_name}_{step}_edges"))
            break

        # Save remnant graph
        save_graph_npz(G_sim, str(rem_dir / f"{net_name}-{algo_name}_{step}_edges"))

        # Save only components with size >= tau
        comps = [G_sim.subgraph(c).copy() for c in nx.connected_components(G_sim)]
        valid = [c for c in comps if c.number_of_nodes() >= tau]
        for idx, comp in enumerate(valid, 1):
            save_graph_npz(comp, str(comp_dir / f"{net_name}-{algo_name}_{step}_{idx}_edges"))


def compute_component_label_single(npz_path_str, tau):
    """
    Worker function: compute brute-force label for one component file.

    Returns
    -------
    (base_name, label_dict) or None
    """
    npz_path = Path(npz_path_str)
    base_name = npz_path.name.replace("_edges.npz", "")
    label_file = npz_path.parent / f"{base_name}_label.json"

    # Incremental mode: reuse existing label if readable
    if label_file.exists():
        try:
            with open(label_file, "r") as f:
                existing = json.load(f)
            return (base_name, existing)
        except Exception:
            pass

    try:
        G = load_graph_from_npz(str(npz_path))
        if G.number_of_nodes() < tau:
            label_data = {
                "critical_threshold": 0.0,
                "removed_nodes": [],
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
            }
        else:
            k, removed = brute_force_min_removal(G, tau)
            n = G.number_of_nodes()
            label_data = {
                "critical_threshold": round(k / n, 6) if n > 0 else 0.0,
                "removed_nodes": [int(x) for x in removed],
                "num_nodes": n,
                "num_edges": G.number_of_edges(),
            }

        with open(label_file, "w") as f:
            json.dump(label_data, f, indent=2)

        return (base_name, label_data)

    except Exception as e:
        print(f"[ERROR] Label computation failed for {base_name}: {e}")
        return None


def aggregate_remnant_labels(rem_dir, comp_dir, net_name, tau):
    """
    Aggregate component labels to remnant labels.

    collapse_distance = (sum of component removal counts) / (remnant node count)
    """
    # Collect all component labels
    comp_labels = {}
    for jf in comp_dir.glob("*_label.json"):
        base = jf.name.replace("_label.json", "")
        with open(jf, "r") as f:
            comp_labels[base] = json.load(f)

    # Aggregate for each remnant
    for rem_file in sorted(rem_dir.glob("*_edges.npz")):
        rem_base = rem_file.name.replace("_edges.npz", "")
        rem_label_file = rem_dir / f"{rem_base}_label.json"

        if rem_label_file.exists():
            continue

        matching_comps = {k: v for k, v in comp_labels.items() if k.startswith(rem_base + "_")}

        total_removal = 0
        all_removed_nodes = []
        for _, comp_label in matching_comps.items():
            ct = comp_label.get("critical_threshold", 0.0)
            cn = comp_label.get("num_nodes", 0)
            total_removal += int(round(ct * cn))
            all_removed_nodes.extend(comp_label.get("removed_nodes", []))

        try:
            G_rem = load_graph_from_npz(str(rem_file))
            rem_nodes = G_rem.number_of_nodes()
            rem_edges = G_rem.number_of_edges()
        except Exception:
            rem_nodes = 0
            rem_edges = 0

        agg_ct = round(total_removal / rem_nodes, 6) if rem_nodes > 0 else 0.0

        label_data = {
            "critical_threshold": agg_ct,
            "removed_nodes": all_removed_nodes,
            "num_nodes": rem_nodes,
            "num_edges": rem_edges,
            "total_removal_count": total_removal,
        }
        with open(rem_label_file, "w") as f:
            json.dump(label_data, f, indent=2)


def process_single_network_attack(net_name, tau, force, workers_for_labels):
    """Run full phase-2 attack pipeline for one (network, tau)."""
    net_dir = DATA_DIR / net_name
    edge_file = net_dir / f"{net_name}_edges.npz"
    if not edge_file.exists():
        print(f"[SKIP] {net_name}: edge file missing")
        return

    tau_dir = net_dir / f"{net_name}-{tau}"
    results_csv = tau_dir / f"{net_name}-{tau}-results.csv"

    if results_csv.exists() and not force:
        print(f"[SKIP] {net_name} tau={tau}: results already exist")
        return

    # Create directory structure
    comp_outer = tau_dir / f"{net_name}-{tau}-Components"
    comp_dir = comp_outer / f"{net_name}-{tau}-Components"
    rem_outer = tau_dir / f"{net_name}-{tau}-Remnants"
    rem_dir = rem_outer / f"{net_name}-{tau}-Remnants"
    comp_dir.mkdir(parents=True, exist_ok=True)
    rem_dir.mkdir(parents=True, exist_ok=True)

    # Load network
    G = load_graph_from_npz(str(edge_file))
    N = G.number_of_nodes()
    print(f"\n[ATTACK] {net_name}: N={N}, E={G.number_of_edges()}, tau={tau}")

    if N < tau:
        print(f"[WARN] {net_name}: N={N} < tau={tau}, skipping")
        return

    # Run 4 attack algorithms
    results_rows = []
    attack_funcs = {
        "DC": attack_dc_static,
        "BC": attack_bc_static,
        "Rand": attack_random,
        "Opt": attack_optimal,
    }

    for algo_name, algo_func in attack_funcs.items():
        print(f"  -> Running {algo_name} ...")
        t0 = time.time()
        seq = algo_func(G, tau)
        elapsed = time.time() - t0
        print(f"     removed {len(seq)} nodes, time {elapsed:.2f}s")

        generate_sequence_files(G, seq, net_name, algo_name, tau, comp_dir, rem_dir)

        results_rows.append({
            "network": net_name,
            "heuristic": algo_name,
            "removals": str(seq),
            "static": "TRUE",
            "threshold": round(tau / N, 6),
            "rem_num": len(seq),
            "network_size": N,
            "critical_threshold": round(len(seq) / N, 6),
            "dismantle_time": round(elapsed, 4),
        })

    # Save attack result CSV
    tau_dir.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(results_csv, index=False)
    print(f"  -> Attack summary saved: {results_csv}")

    # Compute brute-force labels for components
    comp_npz_files = sorted(comp_dir.glob("*_edges.npz"))
    if comp_npz_files:
        print(f"  -> Computing brute-force labels for {len(comp_npz_files)} components...")
        worker_func = partial(compute_component_label_single, tau=tau)
        n_label_workers = min(workers_for_labels, len(comp_npz_files))

        if n_label_workers > 1:
            with ProcessPoolExecutor(max_workers=n_label_workers) as executor:
                futures = {executor.submit(worker_func, str(f)): f for f in comp_npz_files}
                comp_results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"     labels({net_name})"):
                    res = future.result()
                    if res:
                        comp_results.append(res)
        else:
            comp_results = []
            for f in tqdm(comp_npz_files, desc=f"     labels({net_name})"):
                res = worker_func(str(f))
                if res:
                    comp_results.append(res)

        # Save component-label summary CSV
        comp_csv_rows = []
        for base_name, label_data in comp_results:
            cn = label_data.get("num_nodes", 0)
            ct = label_data.get("critical_threshold", 0.0)
            removed = label_data.get("removed_nodes", [])
            comp_csv_rows.append({
                "network": base_name,
                "heuristic": "BruteForce",
                "removals": str(removed),
                "static": "TRUE",
                "threshold": round(tau / cn, 6) if cn > 0 else 0,
                "rem_num": len(removed),
                "network_size": cn,
                "critical_threshold": ct,
                "dismantle_time": 0,
            })

        if comp_csv_rows:
            df_comp = pd.DataFrame(comp_csv_rows)
            comp_csv_path = comp_outer / f"{net_name}-{tau}-Components-results.csv"
            df_comp.to_csv(comp_csv_path, index=False)
            print(f"  -> Component summary saved: {comp_csv_path}")

    # Aggregate remnant labels
    print("  -> Aggregating remnant labels...")
    aggregate_remnant_labels(rem_dir, comp_dir, net_name, tau)
    print(f"  -> Done: {net_name} tau={tau}")


def phase_attack(args):
    """Phase 2: run attacks for one tau on selected/all networks."""
    tau = args.tau
    force = args.force
    workers = args.workers if args.workers else max(1, mp.cpu_count() - 1)

    if args.nets:
        net_names = args.nets
    else:
        net_names = get_all_net_names()

    if not net_names:
        print("[ERROR] No networks found. Please run generate first.")
        return

    print(f"\n{'=' * 60}")
    print(f"[Phase 2: ATTACK] tau={tau}, networks={len(net_names)}, workers={workers}")
    print(f"{'=' * 60}")

    for net_name in net_names:
        process_single_network_attack(net_name, tau, force, workers)


# =============================================================================
# Section 5. Phase 3 — Metric Computation
# =============================================================================

def calc_lcc(G, initial_size):
    """LCC ratio relative to original network size."""
    if G.number_of_nodes() == 0 or initial_size <= 0:
        return 0.0
    comps = list(nx.connected_components(G))
    if not comps:
        return 0.0
    return len(max(comps, key=len)) / initial_size


def calc_natural_connectivity(G, initial_size):
    """Natural connectivity of current remnant graph."""
    if G.number_of_nodes() == 0:
        return 0.0
    try:
        adj = nx.to_numpy_array(G)
        evals = linalg.eigvalsh(adj)
        return float(np.logaddexp.reduce(evals) - np.log(initial_size))
    except Exception:
        return 0.0


def simulate_R(G, mode="random"):
    """
    Simulate attack process and compute R metric:
        R = sum_t LCC_t / N^2

    mode:
      - 'random': random node removal order
      - 'degree': dynamic degree-based removal (DCR)
    """
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return 0.0

    G_sim = G.copy()
    acc = 0.0

    if mode == "random":
        seq = list(np.random.permutation(nodes))
    else:
        seq = None

    for i in range(N):
        if G_sim.number_of_nodes() > 0:
            comps = list(nx.connected_components(G_sim))
            lcc_size = len(max(comps, key=len)) if comps else 0
        else:
            lcc_size = 0

        acc += lcc_size

        if mode == "random":
            node = seq[i]
        elif mode == "degree":
            if G_sim.number_of_nodes() == 0:
                break
            deg = dict(G_sim.degree())
            node = max(deg, key=lambda x: (deg[x], x))
        else:
            break

        if G_sim.has_node(node):
            G_sim.remove_node(node)

    return acc / (N * N)


def compute_metrics_for_remnant(task):
    """
    Worker function for one remnant graph.

    Parameters
    ----------
    task: (npz_path_str, initial_size)

    Returns
    -------
    dict or None
    """
    npz_path_str, initial_size = task
    npz_path = Path(npz_path_str)
    base_name = npz_path.name.replace("_edges.npz", "")

    try:
        G = load_graph_from_npz(npz_path_str)
    except Exception:
        return None

    lcc_val = calc_lcc(G, initial_size)
    nat_conn = calc_natural_connectivity(G, initial_size)
    r_rand = simulate_R(G, mode="random")
    r_dcr = simulate_R(G, mode="degree")

    # Read collapse distance from remnant label
    label_file = npz_path.parent / f"{base_name}_label.json"
    collapse_distance = np.nan

    if label_file.exists():
        try:
            with open(label_file, "r") as f:
                label_data = json.load(f)
            ct = label_data.get("critical_threshold", 0.0)
            nn = label_data.get("num_nodes", 0)
            if "total_removal_count" in label_data:
                collapse_distance = label_data["total_removal_count"] / initial_size
            else:
                collapse_distance = (ct * nn) / initial_size if initial_size > 0 else 0.0
        except Exception:
            pass

    return {
        "base_name": base_name,
        "network_size": G.number_of_nodes(),
        "LCC": round(lcc_val, 6),
        "natural_connectivity": round(nat_conn, 6),
        "R(rand)": round(r_rand, 6),
        "R(DCR)": round(r_dcr, 6),
        "collapse_distance": round(collapse_distance, 6) if not np.isnan(collapse_distance) else np.nan,
    }


def phase_metrics(args):
    """
    Phase 3: compute metrics.
    Supports multiple networks, multiple taus, custom output filename.
    """
    tau_list = args.tau  # None => auto-scan all existing taus
    workers = args.workers if args.workers else max(1, mp.cpu_count() - 1)
    force = args.force
    output_name = args.output if hasattr(args, "output") and args.output else "metrics.csv"

    # Select networks
    if args.nets:
        net_names = args.nets
    else:
        net_names = get_all_net_names()

    if not net_names:
        print("[ERROR] No networks found.")
        return

    # Build (network, tau) pairs
    net_tau_pairs = []
    for net_name in net_names:
        if tau_list:
            for tau in tau_list:
                net_tau_pairs.append((net_name, tau))
        else:
            discovered_taus = get_tau_values_for_net(net_name)
            for tau in discovered_taus:
                net_tau_pairs.append((net_name, tau))

    if not net_tau_pairs:
        print("[ERROR] No (network, tau) pairs found. Please run attack first.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_csv = RESULTS_DIR / output_name

    # Incremental mode: read existing metrics
    if metrics_csv.exists() and not force:
        df_existing = pd.read_csv(metrics_csv)
        existing_keys = set()
        for _, row in df_existing.iterrows():
            key = (
                str(row.get("network", "")),
                str(row.get("algorithm", "")),
                int(row.get("step", -1)),
            )
            existing_keys.add(key)
    else:
        df_existing = pd.DataFrame()
        existing_keys = set()

    tau_summary = {}
    for net_name, tau in net_tau_pairs:
        tau_summary.setdefault(net_name, []).append(tau)

    print(f"\n{'=' * 60}")
    print(f"[Phase 3: METRICS] output={output_name}, workers={workers}")
    for net_name, taus in tau_summary.items():
        print(f"  {net_name}: tau={taus}")
    print(f"  total (network, tau) pairs: {len(net_tau_pairs)}")
    print(f"{'=' * 60}")

    all_new_rows = []

    for net_name, tau in net_tau_pairs:
        net_dir = DATA_DIR / net_name
        edge_file = net_dir / f"{net_name}_edges.npz"
        if not edge_file.exists():
            print(f"[SKIP] {net_name}: edge file missing")
            continue

        # Initial network size
        G_init = load_graph_from_npz(str(edge_file))
        initial_size = G_init.number_of_nodes()
        del G_init

        tau_dir = net_dir / f"{net_name}-{tau}"
        rem_dir = tau_dir / f"{net_name}-{tau}-Remnants" / f"{net_name}-{tau}-Remnants"

        if not rem_dir.exists():
            print(f"[SKIP] {net_name} tau={tau}: remnant dir missing")
            continue

        network_id = f"{net_name}-{tau}"

        # Scan remnant files
        rem_files = sorted(rem_dir.glob("*_edges.npz"))
        if not rem_files:
            print(f"[SKIP] {net_name} tau={tau}: no remnant files")
            continue

        # Build tasks from file names: <net>-<algo>_<step>_edges.npz
        tasks = []
        task_meta = []  # (network_id, algo, step)
        for rf in rem_files:
            base = rf.name.replace("_edges.npz", "")
            suffix = base[len(net_name) + 1:]  # e.g., "DC_5"
            parts = suffix.rsplit("_", 1)
            if len(parts) != 2:
                continue

            algo_name = parts[0]
            try:
                step = int(parts[1])
            except ValueError:
                continue

            key = (network_id, algo_name, step)
            if key in existing_keys:
                continue

            tasks.append((str(rf), initial_size))
            task_meta.append((network_id, algo_name, step))

        if not tasks:
            print(f"[SKIP] {net_name} tau={tau}: all metrics already exist")
            continue

        print(f"  -> {net_name} tau={tau}: computing metrics for {len(tasks)} remnants ...")

        n_w = min(workers, len(tasks))
        if n_w > 1:
            with ProcessPoolExecutor(max_workers=n_w) as executor:
                futures = {executor.submit(compute_metrics_for_remnant, t): i for i, t in enumerate(tasks)}
                results = [None] * len(tasks)
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"     metrics({net_name},tau={tau})"):
                    idx = futures[future]
                    results[idx] = future.result()
        else:
            results = []
            for t in tqdm(tasks, desc=f"     metrics({net_name},tau={tau})"):
                results.append(compute_metrics_for_remnant(t))

        for i, res in enumerate(results):
            if res is None:
                continue
            nid, algo, step = task_meta[i]
            row = {
                "network": nid,
                "algorithm": algo,
                "step": step,
                "network_size": res["network_size"],
                "LCC": res["LCC"],
                "natural_connectivity": res["natural_connectivity"],
                "R(rand)": res["R(rand)"],
                "R(DCR)": res["R(DCR)"],
                "collapse_distance": res["collapse_distance"],
            }
            all_new_rows.append(row)

    # Merge and save
    if all_new_rows:
        df_new = pd.DataFrame(all_new_rows)
        if not df_existing.empty:
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new
    else:
        df_final = df_existing
        print("[INFO] No new metrics to compute.")

    if not df_final.empty:
        df_final = df_final.sort_values(by=["network", "algorithm", "step"]).reset_index(drop=True)
        df_final.to_csv(metrics_csv, index=False)
        print(f"\n[DONE] Metrics saved: {metrics_csv}")
        print(f"       Total rows: {len(df_final)}")
    else:
        print("[INFO] No data to save.")


# =============================================================================
# Section 6. CLI and Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Introduction experiment pipeline: network dismantling",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="phase", help="Pipeline phase")

    # ---- generate ----
    p_gen = subparsers.add_parser("generate", help="Generate one network")
    p_gen.add_argument("--name", required=True, help="Network name (e.g., net1)")
    p_gen.add_argument("--type", required=True, choices=["BA", "ER", "WS", "LFR"], help="Network type")
    p_gen.add_argument("--n", type=int, required=True, help="Number of nodes")
    p_gen.add_argument("--m", type=int, default=3, help="BA parameter m")
    p_gen.add_argument("--avg_degree", type=float, default=4.0, help="ER/LFR average degree")
    p_gen.add_argument("--k", type=int, default=4, help="WS parameter k")
    p_gen.add_argument("--p", type=float, default=0.1, help="WS parameter p")
    p_gen.add_argument("--force", action="store_true", help="Force regeneration")

    # ---- attack ----
    p_atk = subparsers.add_parser("attack", help="Run attack algorithms")
    p_atk.add_argument("--tau", type=int, required=True, help="Disruption target: LCC < tau")
    p_atk.add_argument("--nets", nargs="+", default=None, help="Network names (default: all)")
    p_atk.add_argument("--workers", type=int, default=None, help="Parallel workers")
    p_atk.add_argument("--force", action="store_true", help="Force recomputation")

    # ---- metrics ----
    p_met = subparsers.add_parser("metrics", help="Compute robustness metrics")
    p_met.add_argument(
        "--tau", type=int, nargs="+", default=None,
        help="One or multiple taus (e.g., --tau 5 10). If omitted, scan all existing taus."
    )
    p_met.add_argument("--nets", nargs="+", default=None, help="Network names (default: all)")
    p_met.add_argument("--workers", type=int, default=None, help="Parallel workers")
    p_met.add_argument("--output", type=str, default="metrics.csv", help="Output CSV filename under results/")
    p_met.add_argument("--force", action="store_true", help="Force recomputation")

    # ---- all ----
    p_all = subparsers.add_parser("all", help="Run attack + metrics")
    p_all.add_argument("--tau", type=int, required=True, help="Disruption target tau")
    p_all.add_argument("--nets", nargs="+", default=None, help="Network names (default: all)")
    p_all.add_argument("--workers", type=int, default=None, help="Parallel workers")
    p_all.add_argument("--output", type=str, default="metrics.csv", help="Output CSV filename")
    p_all.add_argument("--force", action="store_true", help="Force recomputation")

    args = parser.parse_args()

    if args.phase is None:
        parser.print_help()
        return

    t_start = time.time()

    if args.phase == "generate":
        phase_generate(args)
    elif args.phase == "attack":
        phase_attack(args)
    elif args.phase == "metrics":
        phase_metrics(args)
    elif args.phase == "all":
        phase_attack(args)
        # phase_metrics expects tau list or None
        args.tau = [args.tau]
        phase_metrics(args)

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.2f} s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
