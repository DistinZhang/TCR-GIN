#!/usr/bin/env python3
"""Strict degree-ordered brute-force search for DC/BC 3-node solutions."""

from __future__ import annotations

import ast
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from itertools import combinations, islice
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "solution_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_ORDER = (
    "remaining nodes in components larger than target_lcc, sorted by degree "
    "descending, then node id ascending; validated on full remaining network"
)


@dataclass(frozen=True)
class Scenario:
    network: str
    attack: str
    heuristic: str
    static: bool
    warning_step: int
    target_lcc: int
    edges_file: Path
    csv_file: Path
    excluded_solution_nodes: tuple[int, ...] = ()


SCENARIOS = [
    Scenario(
        network="transport",
        attack="DC",
        heuristic="degree",
        static=True,
        warning_step=15,
        target_lcc=111,
        edges_file=Path("/root/autodl-tmp/data_trajectory/transport/london_transport_multiplex_aggr_edges.npz"),
        csv_file=Path("/root/autodl-tmp/data_metric/transport_111/transport-results.csv"),
    ),
    Scenario(
        network="transport",
        attack="BC",
        heuristic="betweenness_centrality",
        static=True,
        warning_step=19,
        target_lcc=111,
        edges_file=Path("/root/autodl-tmp/data_trajectory/transport/london_transport_multiplex_aggr_edges.npz"),
        csv_file=Path("/root/autodl-tmp/data_metric/transport_111/transport-results.csv"),
    ),
    Scenario(
        network="power",
        attack="DC",
        heuristic="degree",
        static=True,
        warning_step=98,
        target_lcc=353,
        edges_file=Path("/root/autodl-tmp/data_trajectory/power/power-eris1176_edges.npz"),
        csv_file=Path("/root/autodl-tmp/data_metric/power_353/power-results.csv"),
    ),
    Scenario(
        network="power",
        attack="BC",
        heuristic="betweenness_centrality",
        static=True,
        warning_step=33,
        target_lcc=353,
        edges_file=Path("/root/autodl-tmp/data_trajectory/power/power-eris1176_edges.npz"),
        csv_file=Path("/root/autodl-tmp/data_metric/power_353/power-results.csv"),
        excluded_solution_nodes=(70,),
    ),
]


_ADJ: dict[int, tuple[int, ...]] | None = None
_NODES: tuple[int, ...] | None = None
_TARGET: int | None = None


def init_worker(adj: dict[int, tuple[int, ...]], nodes: tuple[int, ...], target: int) -> None:
    global _ADJ, _NODES, _TARGET
    _ADJ = adj
    _NODES = nodes
    _TARGET = target


def lcc_after_removed(combo: tuple[int, int, int]) -> int:
    assert _ADJ is not None
    assert _NODES is not None
    assert _TARGET is not None

    removed = set(combo)
    seen = set(removed)
    max_size = 0

    for start in _NODES:
        if start in seen:
            continue

        size = 0
        stack = [start]
        seen.add(start)

        while stack:
            node = stack.pop()
            size += 1
            if size > _TARGET:
                return size

            for nbr in _ADJ[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)

        if size > max_size:
            max_size = size

    return max_size


def check_batch(batch: list[tuple[int, int, int]]) -> tuple[int, tuple[int, int, int] | None, tuple[int, int, int] | None, int]:
    best_lcc = 10**12
    best_combo = None

    for combo in batch:
        post_lcc = lcc_after_removed(combo)
        if post_lcc < best_lcc:
            best_lcc = post_lcc
            best_combo = combo
        if post_lcc <= _TARGET:
            return best_lcc, best_combo, combo, len(batch)

    return best_lcc, best_combo, None, len(batch)


def load_graph(edges_file: Path) -> nx.Graph:
    edges = np.load(edges_file)["edges"]
    graph = nx.Graph()
    graph.add_edges_from((int(u), int(v)) for u, v in edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def load_removals(csv_file: Path, heuristic: str, static: bool) -> list[int]:
    df = pd.read_csv(csv_file)
    rows = df[
        (df["heuristic"] == heuristic)
        & (df["static"].astype(str).str.lower() == str(static).lower())
    ]
    if rows.empty:
        raise ValueError(f"No row found for heuristic={heuristic}, static={static} in {csv_file}")

    raw = ast.literal_eval(rows.iloc[0]["removals"])
    return [int(item[1] if isinstance(item, (tuple, list)) else item) for item in raw]


def lcc_size(graph: nx.Graph) -> int:
    return max((len(component) for component in nx.connected_components(graph)), default=0)


def cache_path(network: str, attack: str) -> Path:
    return CACHE_DIR / f"{network}_{attack.lower()}_3node_solution.json"


def batched_combinations(nodes: list[int], batch_size: int):
    iterator = combinations(nodes, 3)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def load_valid_cache(scn: Scenario, rem_graph: nx.Graph, search_space: int) -> dict | None:
    path = cache_path(scn.network, scn.attack)
    if not path.exists():
        return None

    data = json.loads(path.read_text())
    if (
        data.get("warning_step") != scn.warning_step
        or data.get("static") != scn.static
        or data.get("csv_heuristic") != scn.heuristic
        or data.get("search_space") != search_space
        or data.get("candidate_order") != CANDIDATE_ORDER
        or data.get("search_is_strictly_ordered") is not True
        or data.get("excluded_solution_nodes", []) != list(scn.excluded_solution_nodes)
    ):
        return None

    solution = data.get("solution")
    if solution is None:
        if data.get("checked_combinations_approx", 0) >= search_space:
            return data
        return None

    test_graph = rem_graph.copy()
    test_graph.remove_nodes_from(solution)
    if lcc_size(test_graph) <= scn.target_lcc:
        return data

    return None


def search_one(scn: Scenario, workers: int, batch_size: int) -> dict:
    print("\n" + "=" * 78)
    print(f"{scn.network.upper()} | {scn.attack} | step={scn.warning_step} | static={scn.static}")
    print("=" * 78)

    graph = load_graph(scn.edges_file)
    removals = load_removals(scn.csv_file, scn.heuristic, scn.static)
    removed = removals[: scn.warning_step]
    future = removals[scn.warning_step :]

    rem_graph = graph.copy()
    rem_graph.remove_nodes_from(removed)
    large_components = [
        set(component)
        for component in nx.connected_components(rem_graph)
        if len(component) > scn.target_lcc
    ]
    excluded_solution_nodes = set(scn.excluded_solution_nodes)
    candidate_nodes = set().union(*large_components) if large_components else set(rem_graph.nodes())
    candidate_nodes -= excluded_solution_nodes
    ordered = [
        node
        for node, _ in sorted(
            rem_graph.degree(candidate_nodes),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    adj = {node: tuple(rem_graph.neighbors(node)) for node in rem_graph.nodes()}
    all_remaining_nodes = tuple(sorted(rem_graph.nodes()))
    search_space = math.comb(len(ordered), 3)

    print(f"Original: n={graph.number_of_nodes()}, m={graph.number_of_edges()}")
    print(
        f"Warning-step remaining: n={rem_graph.number_of_nodes()}, "
        f"m={rem_graph.number_of_edges()}, LCC={lcc_size(rem_graph)}"
    )
    print(
        "Candidate components larger than target: "
        f"{[len(component) for component in sorted(large_components, key=len, reverse=True)]}"
    )
    print(f"Search candidates: n={len(ordered)}, search space={search_space:,}")
    if excluded_solution_nodes:
        print(f"Excluded solution candidates: {sorted(excluded_solution_nodes)}")
    print(f"Future attacks: {len(future)} nodes")

    cached = load_valid_cache(scn, rem_graph, search_space)
    if cached is not None:
        print(f"Using valid strict cache: {cache_path(scn.network, scn.attack)}")
        return cached

    start = time.time()
    checked = 0
    best_lcc = 10**12
    best_combo = None
    solution = None
    next_batch_to_consume = 0
    buffered = {}
    pending = {}
    batch_iter = enumerate(batched_combinations(ordered, batch_size))
    max_in_flight = max(workers, workers * 4)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(adj, all_remaining_nodes, scn.target_lcc),
    ) as pool:

        def submit_next() -> bool:
            try:
                batch_idx, batch = next(batch_iter)
            except StopIteration:
                return False
            future_obj = pool.submit(check_batch, batch)
            pending[future_obj] = batch_idx
            return True

        for _ in range(max_in_flight):
            if not submit_next():
                break

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future_obj in done:
                batch_idx = pending.pop(future_obj)
                buffered[batch_idx] = future_obj.result()

            while next_batch_to_consume in buffered:
                batch_best_lcc, batch_best_combo, hit, count = buffered.pop(next_batch_to_consume)
                checked += count
                if batch_best_lcc < best_lcc:
                    best_lcc = batch_best_lcc
                    best_combo = batch_best_combo

                if hit is not None:
                    solution = list(hit)
                    print(f"Found first degree-ordered solution: {solution}")
                    break

                next_batch_to_consume += 1
                if checked % max(batch_size * 100, 1) == 0 or checked >= search_space:
                    elapsed = time.time() - start
                    rate = checked / elapsed if elapsed > 0 else 0.0
                    print(f"Checked {checked:,}/{search_space:,} ({rate:,.0f} combos/s), best LCC={best_lcc}")

                submit_next()

            if solution is not None:
                for future_obj in pending:
                    future_obj.cancel()
                break

    elapsed = time.time() - start
    post_lcc = None
    collapse = False
    if solution:
        test_graph = rem_graph.copy()
        test_graph.remove_nodes_from(solution)
        post_lcc = lcc_size(test_graph)
        collapse = post_lcc <= scn.target_lcc

    payload = {
        "network": scn.network,
        "attack_type": scn.attack,
        "csv_heuristic": scn.heuristic,
        "static": scn.static,
        "warning_step": scn.warning_step,
        "target_lcc": scn.target_lcc,
        "original_nodes": graph.number_of_nodes(),
        "original_edges": graph.number_of_edges(),
        "remaining_nodes": rem_graph.number_of_nodes(),
        "remaining_edges": rem_graph.number_of_edges(),
        "remaining_lcc": lcc_size(rem_graph),
        "candidate_components_over_target": [
            len(component) for component in sorted(large_components, key=len, reverse=True)
        ],
        "search_candidate_nodes": len(ordered),
        "search_space": search_space,
        "excluded_solution_nodes": [int(node) for node in scn.excluded_solution_nodes],
        "checked_combinations_approx": checked,
        "solution": solution,
        "solution_post_lcc": post_lcc,
        "solution_collapses": collapse,
        "best_lcc_found": best_lcc if best_combo is not None else None,
        "best_combo_found": list(best_combo) if best_combo is not None else None,
        "future_attacks": [int(node) for node in future],
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_order": CANDIDATE_ORDER,
        "search_is_strictly_ordered": True,
    }
    cache_path(scn.network, scn.attack).write_text(json.dumps(payload, indent=2))

    print(f"Best checked combo: {payload['best_combo_found']} -> LCC {payload['best_lcc_found']}")
    if solution:
        print(f"Solution: {solution} -> post LCC {post_lcc} | collapse={collapse}")
    else:
        print("No 3-node solution found.")
    print(f"Saved cache: {cache_path(scn.network, scn.attack)}")

    return payload


def main() -> None:
    cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 8
    workers = max(8, min(24, cpu // 2))
    batch_size = 5000

    print("=" * 78)
    print(f"DC/BC strict degree-ordered brute-force search | workers={workers} | batch_size={batch_size}")
    print("=" * 78)

    for scn in SCENARIOS:
        search_one(scn, workers=workers, batch_size=batch_size)


if __name__ == "__main__":
    main()
