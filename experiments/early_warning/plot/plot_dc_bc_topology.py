#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Nature-style 2x2 topology figure for DC/BC scenarios.

Main design goals:
1. Four panels in one 2x2 figure.
2. Plot the remaining network after the warning step.
3. Use a fixed full-network layout so that the same node always appears
   at the same position under different attacks.
4. Use shared axis limits for DC/BC of the same network, so the topology
   outline is visually consistent.
5. Export editable SVG/PDF for Illustrator.
"""

from __future__ import annotations

import ast
import json
import pickle
import colorsys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd


# ================================================================
# Paths
# ================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = SCRIPT_DIR / "solution_cache"
LAYOUT_DIR = SCRIPT_DIR / "layout_cache"
OUTPUT_DIR = SCRIPT_DIR / "output"

LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# Scenario configuration
# ================================================================

SCENARIOS = [
    {
        "panel": "a",
        "network": "transport",
        "network_label": "Transport",
        "attack": "DC",
        "heuristic": "degree",
        "static": True,
        "warning_step": 15,
        "target_lcc": 111,
        "edges": Path("/root/autodl-tmp/data_trajectory/transport/london_transport_multiplex_aggr_edges.npz"),
        "results": Path("/root/autodl-tmp/data_metric/transport_111/transport-results.csv"),
        "solution_cache": SOLUTION_DIR / "transport_dc_3node_solution.json",
    },
    {
        "panel": "b",
        "network": "transport",
        "network_label": "Transport",
        "attack": "BC",
        "heuristic": "betweenness_centrality",
        "static": True,
        "warning_step": 19,
        "target_lcc": 111,
        "edges": Path("/root/autodl-tmp/data_trajectory/transport/london_transport_multiplex_aggr_edges.npz"),
        "results": Path("/root/autodl-tmp/data_metric/transport_111/transport-results.csv"),
        "solution_cache": SOLUTION_DIR / "transport_bc_3node_solution.json",
    },
    {
        "panel": "c",
        "network": "power",
        "network_label": "Power grid",
        "attack": "DC",
        "heuristic": "degree",
        "static": True,
        "warning_step": 98,
        "target_lcc": 353,
        "edges": Path("/root/autodl-tmp/data_trajectory/power/power-eris1176_edges.npz"),
        "results": Path("/root/autodl-tmp/data_metric/power_353/power-results.csv"),
        "solution_cache": SOLUTION_DIR / "power_dc_3node_solution.json",
    },
    {
        "panel": "d",
        "network": "power",
        "network_label": "Power grid",
        "attack": "BC",
        "heuristic": "betweenness_centrality",
        "static": True,
        "warning_step": 33,
        "target_lcc": 353,
        "edges": Path("/root/autodl-tmp/data_trajectory/power/power-eris1176_edges.npz"),
        "results": Path("/root/autodl-tmp/data_metric/power_353/power-results.csv"),
        "solution_cache": SOLUTION_DIR / "power_bc_3node_solution.json",
    },
]


# If the solution cache has no explicit future_attacks field, draw only the
# first few attacks after the warning step so future nodes do not dominate.
FUTURE_FALLBACK_N = 3

# Match the typography scale used by generate_scheme3_summary_plots in
# calculate_decision_window.py, which is the line-plot figure this topology
# figure will be paired with.
FS_MAIN = 12
FS_TICK = 11
FS_PANEL_TITLE = 8
FS_PANEL_SUBTITLE = 7


# ================================================================
# Publication style
# ================================================================

def set_publication_style() -> None:
    """Set clean vector-friendly plotting style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.35,

            # Important for Illustrator editing.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",

            # Keep vector paths editable.
            "path.simplify": False,

            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def mm_to_inch(mm: float) -> float:
    return mm / 25.4


# ================================================================
# Data loading
# ================================================================

def as_int_node(x: Any) -> int:
    """Robustly convert cached node objects to int."""
    if isinstance(x, (list, tuple)):
        return int(x[-1])
    return int(x)


def load_graph(edges_file: Path) -> nx.Graph:
    if not edges_file.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_file}")

    data = np.load(edges_file)
    if "edges" not in data:
        raise KeyError(f"'edges' array not found in {edges_file}")

    edges = data["edges"]
    G = nx.Graph()
    G.add_edges_from((int(u), int(v)) for u, v in edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def load_removals(results_file: Path, heuristic: str, static: bool) -> list[int]:
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)

    mask = (
        (df["heuristic"] == heuristic)
        & (df["static"].astype(str).str.lower() == str(static).lower())
    )

    if not mask.any():
        raise ValueError(
            f"No matching row in {results_file} for "
            f"heuristic={heuristic}, static={static}"
        )

    row = df[mask].iloc[0]
    raw = ast.literal_eval(row["removals"])

    removals = []
    for item in raw:
        if isinstance(item, (tuple, list)):
            removals.append(int(item[1]))
        else:
            removals.append(int(item))

    return removals


def load_solution(cache_file: Path) -> tuple[list[int], list[int]]:
    if not cache_file.exists():
        return [], []

    data = json.loads(cache_file.read_text())

    solution = data.get("solution") or []
    future_attacks = data.get("future_attacks") or []

    solution = [as_int_node(n) for n in solution]
    future_attacks = [as_int_node(n) for n in future_attacks]

    # Only keep a cached 3-node solution when it actually collapses the
    # remaining graph to the requested threshold. This prevents stale caches
    # from being rendered as if they were valid solutions.
    target_lcc = data.get("target_lcc")
    solution_post_lcc = data.get("solution_post_lcc")
    solution_collapses = data.get("solution_collapses")

    if solution:
        invalid_cache = False
        if solution_collapses is False:
            invalid_cache = True
        elif (
            target_lcc is not None
            and solution_post_lcc is not None
            and float(solution_post_lcc) > float(target_lcc)
        ):
            invalid_cache = True

        if invalid_cache:
            print(
                f"Cached solution in {cache_file.name} does not satisfy the "
                f"threshold; suppressing solution highlight."
            )
            solution = []

    return solution, future_attacks


# ================================================================
# Layout
# ================================================================

def compute_layout(network: str, G: nx.Graph) -> dict[int, tuple[float, float]]:
    """Compute full-network layout."""
    print(f"Computing layout for {network} network ...")

    if network == "power":
        init_pos = nx.spectral_layout(G, scale=1.0)
        pos = nx.spring_layout(
            G,
            pos=init_pos,
            iterations=450,
            seed=42,
            k=2.6 / np.sqrt(G.number_of_nodes()),
        )
    else:
        pos = nx.spring_layout(
            G,
            iterations=1000,
            seed=42,
            k=2.1 / np.sqrt(G.number_of_nodes()),
        )

    return {int(n): (float(x), float(y)) for n, (x, y) in pos.items()}


def rotate_to_principal_axis(pos: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    """Rotate coordinates so the major axis is approximately horizontal."""
    nodes = list(pos.keys())
    pts = np.array([pos[n] for n in nodes], dtype=float)

    if len(pts) < 3:
        return pos

    center = pts.mean(axis=0)
    centered = pts - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    rot = eigvecs[:, order]

    rotated = centered @ rot

    # Keep orientation deterministic.
    if rotated[:, 0].mean() < 0:
        rotated[:, 0] *= -1

    return {
        node: (float(rotated[i, 0]), float(rotated[i, 1]))
        for i, node in enumerate(nodes)
    }


def compress_tail_layout(
    pos: dict[int, tuple[float, float]],
    G: nx.Graph,
    network: str,
) -> dict[int, tuple[float, float]]:
    """Compress long tails while keeping the global layout consistent.

    This is a display transform only. It preserves the relative ordering of
    nodes and keeps the same node in the same approximate location across
    DC/BC panels, but shortens far-out branches so they do not waste space.
    For power, the transform acts on leaf-to-branch chains so the tail is
    pulled inward without flattening the whole figure.
    """
    if network != "power":
        return pos

    compressed = dict(pos)

    def comp_key(path: list[int]) -> tuple[int, int]:
        # Prefer longer chains, then the more distal leaf.
        return (len(path), -path[-1])

    visited_edges: set[tuple[int, int]] = set()

    for leaf in sorted([n for n in G.nodes() if G.degree(n) == 1]):
        nbrs = list(G.neighbors(leaf))
        if len(nbrs) != 1:
            continue

        path = [leaf]
        prev = None
        cur = leaf

        while True:
            neighbors = [n for n in G.neighbors(cur) if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            edge = tuple(sorted((cur, nxt)))
            if edge in visited_edges:
                break
            path.append(nxt)
            visited_edges.add(edge)
            prev, cur = cur, nxt
            if G.degree(cur) != 2:
                break

        if len(path) < 4:
            continue

        anchor = path[-1]
        if G.degree(anchor) == 1 and len(path) > 2:
            continue

        anchor_pos = np.array(pos[anchor], dtype=float)

        chain_pts = np.array([pos[n] for n in path], dtype=float)
        chain_dists = np.zeros(len(path), dtype=float)
        for i in range(1, len(path)):
            chain_dists[i] = chain_dists[i - 1] + float(np.linalg.norm(chain_pts[i] - chain_pts[i - 1]))

        total = float(chain_dists[-1])
        if total <= 0:
            continue

        # Keep the first part of the chain near the branch, compress the
        # distal tail progressively toward the anchor.
        base_scale = 0.34
        gamma = 1.85

        for node, dist in zip(path[:-1], chain_dists[:-1]):
            t = dist / total
            alpha = 1.0 - (1.0 - base_scale) * (t ** gamma)
            vec = np.array(pos[node], dtype=float) - anchor_pos
            new_xy = anchor_pos + vec * alpha
            compressed[node] = (float(new_xy[0]), float(new_xy[1]))

    return compressed


def compact_disconnected_components(
    pos: dict[int, tuple[float, float]],
    G: nx.Graph,
    network: str,
) -> dict[int, tuple[float, float]]:
    """Pull small disconnected components closer to the main component.

    This keeps each component's internal geometry intact but reduces the
    empty space between components. It is only applied to power, where the
    disconnected tails are visually too spread out after the warning step.
    """
    if network != "power" or G.number_of_nodes() == 0:
        return pos

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if len(components) <= 1:
        return pos

    main_comp = set(components[0])
    main_center = np.mean([pos[n] for n in main_comp if n in pos], axis=0)

    compacted = dict(pos)

    for comp in components[1:]:
        comp = set(comp)
        pts = np.array([pos[n] for n in comp if n in pos], dtype=float)
        if len(pts) == 0:
            continue

        center = pts.mean(axis=0)
        offset = center - main_center
        dist = float(np.linalg.norm(offset))

        # Smaller components should be pulled closer more aggressively.
        if len(comp) == 1:
            shrink = 0.10
        elif len(comp) <= 3:
            shrink = 0.18
        elif len(comp) <= 8:
            shrink = 0.30
        else:
            shrink = 0.48

        if dist > 0:
            new_center = main_center + offset * shrink
            delta = new_center - center
            for n in comp:
                xy = np.array(pos[n], dtype=float) + delta
                compacted[n] = (float(xy[0]), float(xy[1]))

    return compacted


def refine_power_tail_layout(
    pos: dict[int, tuple[float, float]],
    network: str,
) -> dict[int, tuple[float, float]]:
    """Bend the lower power-grid tail upward for a compact panel.

    The transform is deliberately coordinate-based and applied to the shared
    power layout, so DC and BC still use identical node coordinates. It only
    changes display geometry for the long lower branch and the nearby low
    isolated mini-component; graph topology and component membership are not
    touched.
    """
    if network != "power":
        return pos

    adjusted = dict(pos)

    # The low mini-component appears as an isolated point in the manuscript
    # panels. Move it upward first, then bend the distal tail toward it.
    island_nodes = [
        n for n, (x, y) in pos.items()
        if -0.09 <= x <= 0.08 and -0.32 <= y <= -0.18
    ]

    if island_nodes:
        island_pts = np.array([pos[n] for n in island_nodes], dtype=float)
        island_center = island_pts.mean(axis=0)
        island_target = island_center + np.array([0.030, 0.145])

        for n in island_nodes:
            xy = np.array(pos[n], dtype=float)
            new_xy = island_target + (xy - island_center) * 0.72
            adjusted[n] = (float(new_xy[0]), float(new_xy[1]))
    else:
        island_target = np.array([0.0, -0.13], dtype=float)

    # Lower right tail: rotate and compress its distal end toward the moved
    # mini-component while leaving the attachment to the main body stable.
    tail_nodes = [
        n for n, (x, y) in pos.items()
        if x > 0.13 and y < -0.07
    ]
    if len(tail_nodes) < 8:
        return adjusted

    tail_pts = np.array([pos[n] for n in tail_nodes], dtype=float)
    top_y = float(tail_pts[:, 1].max())
    bottom_y = float(tail_pts[:, 1].min())
    if top_y <= bottom_y:
        return adjusted

    top_cut = np.percentile(tail_pts[:, 1], 84)
    bottom_cut = np.percentile(tail_pts[:, 1], 12)
    top_anchor = tail_pts[tail_pts[:, 1] >= top_cut].mean(axis=0)
    bottom_center = tail_pts[tail_pts[:, 1] <= bottom_cut].mean(axis=0)

    old_vec = bottom_center - top_anchor
    target_end = island_target + np.array([0.065, -0.005])
    new_vec = target_end - top_anchor

    old_len = float(np.linalg.norm(old_vec))
    new_len = float(np.linalg.norm(new_vec))
    if old_len <= 1e-9 or new_len <= 1e-9:
        return adjusted

    old_angle = float(np.arctan2(old_vec[1], old_vec[0]))
    new_angle = float(np.arctan2(new_vec[1], new_vec[0]))
    theta = new_angle - old_angle
    scale = min(0.38, max(0.12, new_len / old_len))

    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )

    for n in tail_nodes:
        xy = np.array(pos[n], dtype=float)
        t = np.clip((top_y - xy[1]) / (top_y - bottom_y), 0.0, 1.0)

        transformed = top_anchor + scale * (rot @ (xy - top_anchor))

        # Nonlinear blending keeps the branch root attached but bends the
        # visually dominant distal tail toward the moved mini-component.
        blend = float(t ** 0.62)
        new_xy = xy * (1.0 - blend) + transformed * blend

        distal_pull = float(t ** 2.4)
        if distal_pull > 0:
            distal_target = target_end + (xy - bottom_center) * 0.12
            pull = 0.68 * distal_pull
            new_xy = new_xy * (1.0 - pull) + distal_target * pull

        adjusted[n] = (float(new_xy[0]), float(new_xy[1]))

    return adjusted


def load_or_create_layout(network: str, G: nx.Graph) -> dict[int, tuple[float, float]]:
    """Load one fixed layout per network.

    Important:
    The layout is computed on the original full graph, not on attacked LCCs.
    Therefore the same node has the same coordinate in DC and BC panels.
    """
    path = LAYOUT_DIR / f"{network}_fixed_full_layout.pkl"

    if path.exists():
        with path.open("rb") as f:
            pos = pickle.load(f)
        pos = compress_tail_layout(pos, G, network)
        pos = compact_disconnected_components(pos, G, network)
        return refine_power_tail_layout(pos, network)

    pos = compute_layout(network, G)
    pos = rotate_to_principal_axis(pos)

    with path.open("wb") as f:
        pickle.dump(pos, f)

    pos = compress_tail_layout(pos, G, network)
    pos = compact_disconnected_components(pos, G, network)
    return refine_power_tail_layout(pos, network)


# ================================================================
# Scenario preparation
# ================================================================

def prepare_scenario(scn: dict) -> dict:
    G = load_graph(scn["edges"])
    removals = load_removals(scn["results"], scn["heuristic"], scn["static"])

    removed = removals[: scn["warning_step"]]
    future_from_csv = removals[scn["warning_step"] :]

    R = G.copy()
    R.remove_nodes_from(removed)

    plot_graph = R.copy()

    pos = load_or_create_layout(scn["network"], G)

    solution_cached, future_cached = load_solution(scn["solution_cache"])

    if future_cached:
        future = future_cached
    else:
        future = future_from_csv[:FUTURE_FALLBACK_N]

    plot_nodes = set(plot_graph.nodes())

    solution = [n for n in solution_cached if n in plot_nodes]
    future = [n for n in future if n in plot_nodes]

    return {
        "G": G,
        "R": R,
        "plot_graph": plot_graph,
        "pos": pos,
        "solution": solution,
        "future": future,
        "remaining_size": plot_graph.number_of_nodes(),
        "remaining_edges": plot_graph.number_of_edges(),
        "removed_count": len(removed),
    }


# ================================================================
# Shared axis limits
# ================================================================

def get_limits_for_nodes(
    pos: dict[int, tuple[float, float]],
    nodes: set[int],
    pad_fraction: float = 0.075,
) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = np.array([pos[n] for n in nodes if n in pos], dtype=float)

    if len(pts) == 0:
        return (-1.0, 1.0), (-1.0, 1.0)

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    dx = xmax - xmin
    dy = ymax - ymin

    if dx <= 0:
        dx = 1.0
    if dy <= 0:
        dy = 1.0

    xpad = dx * pad_fraction
    ypad = dy * pad_fraction

    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def make_shared_limits(
    scenarios: list[dict],
    common_frame: tuple[float, float],
    outer_pad_fraction: float = 0.06,
) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Use one viewport per network family.

    The small outer pad keeps node circles and halos fully inside the axes
    so the border never clips a marker.
    """
    limits = {}

    target_w, target_h = common_frame
    x_half = target_w / 2.0 * (1.0 + outer_pad_fraction)
    y_half = target_h / 2.0 * (1.0 + outer_pad_fraction)
    xlim = (-x_half, x_half)
    ylim = (-y_half, y_half)

    for network in sorted(set(s["network"] for s in scenarios)):
        limits[network] = (xlim, ylim)

    return limits


def fit_layout_to_frame(
    pos: dict[int, tuple[float, float]],
    target_w: float,
    target_h: float,
    content_scale: float = 0.90,
) -> dict[int, tuple[float, float]]:
    """Affine-fit a layout to a common display frame.

    This is a presentation transform only. It lets different network families
    occupy the same physical width/height in the panel while preserving the
    same within-family coordinates across attack scenarios.
    """
    if not pos:
        return pos

    pts = np.array(list(pos.values()), dtype=float)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)

    width = xmax - xmin
    height = ymax - ymin
    if width <= 0:
        width = 1.0
    if height <= 0:
        height = 1.0

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    sx = target_w / width
    sy = target_h / height
    sx *= content_scale
    sy *= content_scale

    return {
        n: (
            float((x - cx) * sx),
            float((y - cy) * sy),
        )
        for n, (x, y) in pos.items()
    }


def normalize_family_frames(
    scenarios: list[dict],
    prepared: list[dict],
    pad: float = 0.06,
    content_scale: float = 0.90,
) -> tuple[float, float]:
    """Normalize transport and power layouts to one common display frame."""
    family_frames: dict[str, tuple[float, float]] = {}

    for network in sorted(set(s["network"] for s in scenarios)):
        union_nodes: set[int] = set()
        pos_ref = None

        for scn, data in zip(scenarios, prepared):
            if scn["network"] != network:
                continue
            union_nodes.update(data["plot_graph"].nodes())
            pos_ref = data["pos"]

        if pos_ref is None or not union_nodes:
            continue

        pts = np.array([pos_ref[n] for n in union_nodes if n in pos_ref], dtype=float)
        if len(pts) == 0:
            continue

        width = float(pts[:, 0].max() - pts[:, 0].min())
        height = float(pts[:, 1].max() - pts[:, 1].min())
        family_frames[network] = (width, height)

    if not family_frames:
        return 1.0, 1.0

    target_w = max(w for w, _ in family_frames.values()) * (1.0 + pad)
    target_h = max(h for _, h in family_frames.values()) * (1.0 + pad)

    for data in prepared:
        data["pos"] = fit_layout_to_frame(
            data["pos"],
            target_w=target_w,
            target_h=target_h,
            content_scale=content_scale,
        )

    return target_w, target_h


# ================================================================
# Drawing helpers
# ================================================================

def set_gid(artist, gid: str):
    """Assign SVG group id when possible."""
    if artist is not None and hasattr(artist, "set_gid"):
        artist.set_gid(gid)
    return artist


def draw_nodes(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict[int, tuple[float, float]],
    nodelist: list[int],
    node_color: str,
    node_size: float,
    edgecolors: str,
    linewidths: float,
    alpha: float,
    gid: str,
):
    if not nodelist:
        return None

    artist = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=sorted(nodelist),
        ax=ax,
        node_color=node_color,
        node_size=node_size,
        edgecolors=edgecolors,
        linewidths=linewidths,
        alpha=alpha,
    )
    set_gid(artist, gid)
    return artist


def spread_overlapping_nodes(
    pos: dict[int, tuple[float, float]],
    nodelist: list[int],
    radius: float,
) -> dict[int, tuple[float, float]]:
    """Return display-only positions with exactly overlapping markers separated.

    The network layout itself remains fixed. This helper only offsets high-
    light markers that would otherwise be indistinguishable in the exported
    figure, such as two selected nodes with identical layout coordinates.
    """
    adjusted = dict(pos)
    groups: dict[tuple[float, float], list[int]] = {}

    for node in nodelist:
        if node not in pos:
            continue
        x, y = pos[node]
        key = (round(float(x), 10), round(float(y), 10))
        groups.setdefault(key, []).append(node)

    for nodes in groups.values():
        if len(nodes) <= 1:
            continue

        nodes = sorted(nodes)
        cx, cy = pos[nodes[0]]
        start_angle = np.pi / 2.0
        for idx, node in enumerate(nodes):
            angle = start_angle + 2.0 * np.pi * idx / len(nodes)
            adjusted[node] = (
                float(cx + radius * np.cos(angle)),
                float(cy + radius * np.sin(angle)),
            )

    return adjusted


def draw_edges(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict[int, tuple[float, float]],
    edgelist: list[tuple[int, int]],
    edge_color: str,
    width: float,
    alpha: float,
    gid: str,
):
    if not edgelist:
        return None

    artist = nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edgelist,
        ax=ax,
        edge_color=edge_color,
        width=width,
        alpha=alpha,
    )
    set_gid(artist, gid)
    return artist


def get_component_palette(n: int) -> list[str]:
    """Generate `n` distinct component colors with no reuse.

    Colors stay away from orange/red so they do not compete with future
    attack nodes or the 3-node solution. The leading colors are hand-tuned
    muted cool hues; fallback colors are generated only if a panel has more
    visible components than the curated list.
    """
    if n <= 0:
        return []

    curated = [
        "#3a59a6", "#c46cda", "#6cc0da", "#8b2db4",
        "#716cda", "#663776", "#23238b", "#2d93b4",
        "#a05fb4", "#5243d0", "#5f9fb4", "#376776",
        "#352db4", "#613aa6", "#b443d0", "#437cd0",
        "#488099", "#43aed0", "#828cc4", "#5f67b4",
        "#9b6cda", "#373e76", "#4a238b", "#7c43d0",
        "#834899", "#b082c4", "#23528b", "#82b0c4",
        "#6c9cda", "#76238b", "#2d72b4", "#805fb4",
        "#23728b", "#435ed0", "#4d4899", "#6851c2",
        "#5f83b4", "#2d46b4", "#9743d0", "#712db4",
        "#8777cf", "#4394d0", "#6c84da", "#486899",
    ]

    if n <= len(curated):
        return curated[:n]

    def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
        c = hex_color.lstrip("#")
        return (
            int(c[0:2], 16) / 255.0,
            int(c[2:4], 16) / 255.0,
            int(c[4:6], 16) / 255.0,
        )

    candidates: list[tuple[str, tuple[float, float, float]]] = [
        (color, hex_to_rgb(color)) for color in curated
    ]

    fallback_hues = np.concatenate(
        [
            np.linspace(0.30, 0.43, 48, endpoint=True),
            np.linspace(0.47, 0.62, 64, endpoint=True),
            np.linspace(0.66, 0.78, 48, endpoint=True),
        ]
    )
    for h in fallback_hues:
        for s in [0.40, 0.52, 0.64]:
            for l in [0.34, 0.46, 0.58, 0.70]:
                r, g, b = colorsys.hls_to_rgb(float(h), float(l), float(s))
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    int(round(r * 255)),
                    int(round(g * 255)),
                    int(round(b * 255)),
                )
                candidates.append((hex_color, (r, g, b)))

    def dist2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2

    selected: list[str] = list(curated)
    selected_rgb: list[tuple[float, float, float]] = [hex_to_rgb(color) for color in curated]

    remaining = [c for c in candidates if c[0] not in selected]
    while len(selected) < n and remaining:
        best_i = 0
        best_score = -1.0
        for i, (_, rgb) in enumerate(remaining):
            score = min(dist2(rgb, existing) for existing in selected_rgb)
            if score > best_score:
                best_score = score
                best_i = i
        color, rgb = remaining.pop(best_i)
        selected.append(color)
        selected_rgb.append(rgb)

    return selected[:n]


def build_component_color_map(
    G: nx.Graph,
) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
    """Map nodes and edges to component colors for visible background nodes."""
    components = sorted(
        (set(c) for c in nx.connected_components(G)),
        key=len,
        reverse=True,
    )
    palette = get_component_palette(len(components))

    node_to_color: dict[int, str] = {}
    edge_to_color: dict[tuple[int, int], str] = {}

    for idx, comp in enumerate(components):
        color = palette[idx % len(palette)]
        for node in comp:
            node_to_color[node] = color
        for u, v in G.subgraph(comp).edges():
            edge_to_color[(u, v)] = color
            edge_to_color[(v, u)] = color

    return node_to_color, edge_to_color


def lighten_hex(hex_color: str, mix: float = 0.35) -> str:
    """Mix a hex color with white by `mix`."""
    c = hex_color.lstrip("#")
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    r = int(round(r + (255 - r) * mix))
    g = int(round(g + (255 - g) * mix))
    b = int(round(b + (255 - b) * mix))
    return f"#{r:02x}{g:02x}{b:02x}"


# ================================================================
# Draw panel
# ================================================================

def draw_panel(
    ax: plt.Axes,
    scn: dict,
    data: dict,
    shared_lim: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    G: nx.Graph = data["plot_graph"]
    pos = data["pos"]

    ax.set_aspect("equal")
    ax.set_anchor("C")
    ax.axis("off")
    ax.set_facecolor("white")

    if G.number_of_nodes() == 0:
        ax.text(
            0.5,
            0.5,
            "No nodes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=7,
        )
        return

    # -------------------------
    # Visual parameters
    # -------------------------
    if scn["network"] == "transport":
        edge_width = 0.29
        edge_alpha = 0.23

        normal_size = 7.0
        normal_alpha = 0.70

        future_halo_size = 64
        future_core_size = 24

        solution_halo_size = 78
        solution_core_size = 29

    else:
        edge_width = 0.13
        edge_alpha = 0.21

        normal_size = 7.0
        normal_alpha = 0.70

        future_halo_size = 64
        future_core_size = 24

        solution_halo_size = 78
        solution_core_size = 29

    # Nature-style muted palette.
    edge_color = "#b9c4cf"
    normal_edge = "#ffffff"

    future_halo = "#f6d27a"
    future_core = "#c98a12"
    future_edge = "#7a4c06"

    solution_halo = "#eda0a8"
    solution_core = "#bf2838"
    solution_edge = "#6b1120"

    future_set = set(data["future"])
    solution_set = set(data["solution"])
    highlight_set = future_set | solution_set

    # Connected components are computed on the current remaining network G.
    # We do not recompute components after excluding future/solution nodes for
    # highlighting; those nodes are overlaid later, but the background
    # component partition still comes from the current network state.
    components = sorted(
        (sorted(component) for component in nx.connected_components(G)),
        key=lambda nodes: (-len(nodes), nodes[0]),
    )
    background_palette = get_component_palette(len(components))

    edge_halo_alpha = 0.105
    node_halo_alpha = 0.14
    node_halo_scale = 1.95
    highlight_spread_radius = 0.026
    future_pos = spread_overlapping_nodes(pos, data["future"], highlight_spread_radius)
    solution_pos = spread_overlapping_nodes(pos, data["solution"], highlight_spread_radius)

    # -------------------------
    # Edges
    # -------------------------
    for i, comp_nodes in enumerate(components):
        comp_color = background_palette[i]
        comp_subgraph = G.subgraph(comp_nodes)
        comp_edges = list(comp_subgraph.edges())

        if comp_edges:
            draw_edges(
                ax=ax,
                G=G,
                pos=pos,
                edgelist=comp_edges,
                edge_color=lighten_hex(comp_color, 0.62),
                width=edge_width * 2.15,
                alpha=edge_halo_alpha,
                gid=f"panel_{scn['panel']}_background_edges_halo_{i}",
            )
            draw_edges(
                ax=ax,
                G=G,
                pos=pos,
                edgelist=comp_edges,
                edge_color=comp_color,
                width=edge_width,
                alpha=min(0.78, edge_alpha + 0.28),
                gid=f"panel_{scn['panel']}_background_edges_{i}",
            )

    # -------------------------
    # Ordinary nodes
    # -------------------------
    for i, comp_nodes in enumerate(components):
        display_nodes = [n for n in comp_nodes if n not in highlight_set]
        if not display_nodes:
            continue

        draw_nodes(
            ax=ax,
            G=G,
            pos=pos,
            nodelist=display_nodes,
            node_color=lighten_hex(background_palette[i], 0.55),
            node_size=normal_size * node_halo_scale,
            edgecolors="none",
            linewidths=0.0,
            alpha=node_halo_alpha,
            gid=f"panel_{scn['panel']}_background_nodes_halo_{i}",
        )
        draw_nodes(
            ax=ax,
            G=G,
            pos=pos,
            nodelist=display_nodes,
            node_color=background_palette[i],
            node_size=normal_size,
            edgecolors=normal_edge,
            linewidths=0.20,
            alpha=0.98,
            gid=f"panel_{scn['panel']}_background_nodes_{i}",
        )

    # -------------------------
    # Future attack nodes
    # -------------------------
    draw_nodes(
        ax=ax,
        G=G,
        pos=future_pos,
        nodelist=data["future"],
        node_color=future_halo,
        node_size=future_halo_size,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.34,
        gid=f"panel_{scn['panel']}_future_attack_halo",
    )

    draw_nodes(
        ax=ax,
        G=G,
        pos=future_pos,
        nodelist=data["future"],
        node_color=future_core,
        node_size=future_core_size,
        edgecolors=future_edge,
        linewidths=0.46,
        alpha=0.98,
        gid=f"panel_{scn['panel']}_future_attack_nodes",
    )

    # -------------------------
    # Solution nodes
    # -------------------------
    draw_nodes(
        ax=ax,
        G=G,
        pos=solution_pos,
        nodelist=data["solution"],
        node_color=solution_halo,
        node_size=solution_halo_size,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.34,
        gid=f"panel_{scn['panel']}_solution_halo",
    )

    draw_nodes(
        ax=ax,
        G=G,
        pos=solution_pos,
        nodelist=data["solution"],
        node_color=solution_core,
        node_size=solution_core_size,
        edgecolors=solution_edge,
        linewidths=0.56,
        alpha=0.99,
        gid=f"panel_{scn['panel']}_solution_nodes",
    )

    # -------------------------
    # Shared limits
    # -------------------------
    xlim, ylim = shared_lim
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # -------------------------
    # Panel label and title
    # -------------------------
    ax.text(
        0.012,
        0.988,
        scn["panel"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_PANEL_TITLE,
        fontweight="bold",
        color="#111111",
    )

    title = f"{scn['network_label']} | {scn['attack']}"
    subtitle = f"Remaining = {data['remaining_size']}, warning step = {scn['warning_step']}"

    ax.text(
        0.092,
        0.988,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_MAIN,
        fontweight="bold",
        color="#111111",
    )

    ax.text(
        0.092,
        0.918,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_PANEL_SUBTITLE,
        color="#444444",
    )


# ================================================================
# Legend
# ================================================================

def build_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#7f98ae",
            markeredgecolor="#ffffff",
            markeredgewidth=0.4,
            markersize=6.0,
            label="Background components",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#d28a18",
            markeredgecolor="#704707",
            markeredgewidth=0.6,
            markersize=6.4,
            label="Future attack nodes",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#c92535",
            markeredgecolor="#641018",
            markeredgewidth=0.6,
            markersize=6.4,
            label="3-node solution",
        ),
        Line2D(
            [0],
            [0],
            color="#aeb7c2",
            linewidth=1.1,
            alpha=0.65,
            label="Edges",
        ),
    ]


# ================================================================
# Main
# ================================================================

def main() -> None:
    set_publication_style()

    print("=" * 80)
    print("Nature-style 2x2 LCC topology figure")
    print("=" * 80)

    prepared = []
    for scn in SCENARIOS:
        data = prepare_scenario(scn)
        prepared.append(data)

        print(
            f"{scn['panel']}) {scn['network']} {scn['attack']}: "
            f"remaining nodes={data['remaining_size']}, "
            f"remaining edges={data['remaining_edges']}, "
            f"future={data['future']}, "
            f"solution={data['solution']}"
        )

    common_frame = normalize_family_frames(SCENARIOS, prepared, pad=0.08, content_scale=0.90)
    shared_limits = make_shared_limits(SCENARIOS, common_frame, outer_pad_fraction=0.06)

    # Double-column Nature-style width: around 183 mm.
    # Height is slightly increased to make edges and nodes readable.
    fig_w = mm_to_inch(183)
    fig_h = mm_to_inch(168)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
    )

    for ax, scn, data in zip(axes.ravel(), SCENARIOS, prepared):
        shared_lim = shared_limits[scn["network"]]
        draw_panel(ax, scn, data, shared_lim)

    handles = build_legend()

    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FS_MAIN,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=0.9,
        labelspacing=0.35,
        bbox_to_anchor=(0.5, 0.010),
    )
    set_gid(legend, "global_legend")

    fig.subplots_adjust(
        left=0.030,
        right=0.985,
        top=0.982,
        bottom=0.118,
        wspace=0.050,
        hspace=0.070,
    )

    out_base = OUTPUT_DIR / "dc_bc_lcc_topology_2x2"

    # SVG is recommended for Illustrator editing.
    fig.savefig(out_base.with_suffix(".svg"), format="svg")

    # PDF is also vector-based, suitable for submission.
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf")

    # PNG only for quick preview.
    fig.savefig(out_base.with_suffix(".png"), format="png", dpi=600)

    plt.close(fig)

    print("\nSaved:")
    print(f"  {out_base.with_suffix('.svg')}")
    print(f"  {out_base.with_suffix('.pdf')}")
    print(f"  {out_base.with_suffix('.png')}")
    print("\nFor Illustrator editing, use the SVG file first.")
    print("Done.")


if __name__ == "__main__":
    main()
