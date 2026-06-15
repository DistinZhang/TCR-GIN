#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Network loading and validation utilities."""

from typing import Dict, Tuple

import networkx as nx
import numpy as np


class NetworkLoader:
    """Load and validate network data from NPZ edge-list files."""

    @staticmethod
    def _canonical_node_order(G: nx.Graph) -> list:
        try:
            return sorted(G.nodes())
        except TypeError:
            return list(G.nodes())

    @staticmethod
    def _is_zero_based_natural_nodes(nodes: list) -> bool:
        return all(isinstance(node, (int, np.integer)) for node in nodes) and list(nodes) == list(range(len(nodes)))

    @staticmethod
    def relabel_to_zero_based(G: nx.Graph) -> Tuple[nx.Graph, Dict]:
        """Relabel nodes to contiguous integers 0..N-1 and return mapping metadata."""
        nodes = NetworkLoader._canonical_node_order(G)
        already_canonical = NetworkLoader._is_zero_based_natural_nodes(nodes)
        if already_canonical:
            to_internal = {int(node): int(node) for node in nodes}
            to_original = {int(node): int(node) for node in nodes}
            relabeled = G.copy()
        else:
            to_internal = {node: idx for idx, node in enumerate(nodes)}
            to_original = {idx: node for node, idx in to_internal.items()}
            relabeled = nx.relabel_nodes(G, to_internal, copy=True)

        return relabeled, {
            "relabelled": not already_canonical,
            "to_internal": to_internal,
            "to_original": to_original,
            "original_nodes_preview": nodes[:10],
        }

    @staticmethod
    def load_from_npz(file_path: str) -> Tuple[nx.Graph, Dict]:
        """Load a NetworkX graph from a `.npz` edge-list file."""
        try:
            with np.load(file_path, allow_pickle=True) as loader:
                if "edges" in loader:
                    edges = loader["edges"]
                elif "data" in loader:
                    edges = loader["data"]
                elif "edge_index" in loader:
                    edges = loader["edge_index"]
                else:
                    keys = list(loader.keys())
                    if not keys:
                        raise ValueError("The NPZ file is empty.")
                    edges = loader[keys[0]]
        except Exception as exc:
            raise ValueError(f"Could not read the file: {exc}") from exc

        if edges.ndim == 2 and edges.shape[0] == 2 and edges.shape[1] != 2:
            edges = edges.T

        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(
                f"Invalid edge array format.\n"
                f"Current shape: {edges.shape}\n"
                f"Expected shape: (E, 2) or edge_index format (2, E).\n\n"
                f"Example:\n"
                f"[[0, 1],\n"
                f" [1, 2],\n"
                f" [2, 3],\n"
                f" ...]\n\n"
                f"Conversion example:\n"
                f"edges = np.array([[source1, target1], [source2, target2], ...])\n"
                f"np.savez('network_edges.npz', edges=edges)"
            )

        G = nx.Graph()
        G.add_edges_from(edges)
        G.remove_edges_from(nx.selfloop_edges(G))
        G, node_mapping = NetworkLoader.relabel_to_zero_based(G)

        metadata = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "avg_degree": 2 * G.number_of_edges() / G.number_of_nodes()
            if G.number_of_nodes() > 0
            else 0,
            "density": nx.density(G),
            "is_connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
            "node_mapping": node_mapping,
            "node_ids_relabelled": node_mapping["relabelled"],
        }

        components = list(nx.connected_components(G))
        metadata["num_components"] = len(components)
        metadata["largest_component_size"] = len(max(components, key=len)) if components else 0
        metadata["largest_component_ratio"] = (
            metadata["largest_component_size"] / metadata["num_nodes"]
            if metadata["num_nodes"] > 0
            else 0
        )

        return G, metadata

    @staticmethod
    def validate_network(G: nx.Graph) -> bool:
        """Validate that a graph is usable for analysis."""
        if G.number_of_nodes() == 0:
            raise ValueError("The network is empty.")

        if G.number_of_edges() == 0:
            raise ValueError("The network has no edges.")

        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            largest = len(max(components, key=len))
            print(
                f"Warning: the network is disconnected with {len(components)} components; "
                f"the largest component has {largest} nodes."
            )

        return True

    @staticmethod
    def get_node_mapping(G: nx.Graph) -> Dict:
        """Create mappings between original node IDs and integer indices."""
        nodes = NetworkLoader._canonical_node_order(G)
        to_index = {node: i for i, node in enumerate(nodes)}
        to_original = {i: node for i, node in enumerate(nodes)}
        return {
            "to_index": to_index,
            "to_original": to_original,
            "nodes_list": nodes,
        }
