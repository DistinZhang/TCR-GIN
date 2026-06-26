#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attack Simulator Module - FINAL CORRECTED VERSION

Implements the attack simulation logic used by the early-warning scripts.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple
import csv
from io import StringIO

try:
    import torch
except Exception:
    torch = None
Data = None


def _ensure_torch_geometric():
    """Import PyG Data lazily so page navigation does not pay this cost."""
    global Data
    if torch is None:
        raise RuntimeError("torch is not installed; TCR-GIN inference is unavailable.")
    if Data is None:
        try:
            from torch_geometric.data import Data as PyGData
            Data = PyGData
        except Exception as exc:
            raise RuntimeError("torch_geometric is not installed; TCR-GIN inference is unavailable.") from exc
    return Data


FEATURE_NAMES = [
    "degree",
    "clustering",
    "kcore",
    "average_neighbor_degree",
    "pagerank",
    "betweenness_centrality",
    "eigenvector_centrality",
]


class AttackSimulator:
    """Simulate network attacks and track metrics."""

    def __init__(
        self,
        G: nx.Graph,
        attack_sequence: List,
        model=None,
        scale_factor=1.0,
        collapse_target_ratio: float = 0.0,
        compute_natural_connectivity: bool = True,
        compute_r_value: bool = True,
    ):
        """
        Initialize attack simulator.

        Args:
            G: Original network (will be copied)
            attack_sequence: List of nodes to attack in order
            model: TCR-GIN model for collapse distance prediction (optional)
            scale_factor: Internal compatibility parameter for old callers.
        """
        self.original_G = G.copy()
        self.current_G = G.copy()
        self.attack_sequence = attack_sequence
        self.step = 0
        self.history = []
        self.model = model
        self.scale_factor = scale_factor
        self.model_error = None
        self.collapse_target_ratio = float(collapse_target_ratio or 0.0)
        self.compute_natural_connectivity = bool(compute_natural_connectivity)
        self.compute_r_value = bool(compute_r_value)

        # Store original metrics
        self.original_nodes = self.original_G.number_of_nodes()
        self.original_edges = self.original_G.number_of_edges()

        # Record initial state (step=0)
        initial_metrics = self.calculate_metrics()
        initial_metrics['step'] = 0
        initial_metrics['node_removed'] = None
        initial_metrics['skipped'] = False
        self.history.append(initial_metrics)
        self.step = 1  # Next attack will be step=1

        # Calculate initial collapse distance
        initial_dc = initial_metrics.get('collapse_distance')
        self.initial_collapse_distance = (
            float(initial_dc)
            if initial_dc is not None and np.isfinite(initial_dc)
            else None
        )

    def reset(self):
        """Reset simulation to initial state."""
        self.current_G = self.original_G.copy()
        self.step = 0
        self.history = []

        # Re-record initial state
        initial_metrics = self.calculate_metrics()
        initial_metrics['step'] = 0
        initial_metrics['node_removed'] = None
        initial_metrics['skipped'] = False
        self.history.append(initial_metrics)
        self.step = 1
        initial_dc = initial_metrics.get('collapse_distance')
        self.initial_collapse_distance = (
            float(initial_dc)
            if initial_dc is not None and np.isfinite(initial_dc)
            else None
        )

    @staticmethod
    def _without_isolates(G: nx.Graph) -> nx.Graph:
        """Return the residual graph used by experiment remnant files.

        The experiment pipeline saves only edge lists for remnants, so isolated
        nodes disappear when those files are loaded for robustness metrics.
        """
        analysis_G = G.copy()
        if analysis_G.number_of_nodes() > 0:
            analysis_G.remove_nodes_from(list(nx.isolates(analysis_G)))
        return analysis_G

    def analysis_graph(self) -> nx.Graph:
        """Current residual graph with isolated nodes removed."""
        return self._without_isolates(self.current_G)

    def predict_collapse_distance(
        self,
        graph: Optional[nx.Graph] = None,
        components: Optional[List[set]] = None,
        lcc_nodes: Optional[int] = None,
    ) -> float:
        """
        Predict Collapse Distance using the TCR-GIN model.

        The web app follows the experiment scripts:
        1. Predict the current residual graph critical threshold.
        2. If the residual graph has multiple connected components, predict each
           component whose size is still above the collapse target and aggregate
           the component predictions additively.
        3. Normalize the result to the initial network scale.

        Returns:
            Collapse Distance on the initial-network scale.
        """
        graph = self.analysis_graph() if graph is None else graph
        current_nodes = graph.number_of_nodes()

        if current_nodes == 0:
            return 0.0

        if self.model is None:
            return np.nan

        if self.collapse_target_ratio > 0:
            collapse_target_nodes = self.collapse_target_ratio * self.original_nodes
            if lcc_nodes is None:
                if components is None:
                    components = list(nx.connected_components(graph))
                lcc_nodes = len(max(components, key=len)) if components else 0
            if lcc_nodes <= collapse_target_nodes:
                return 0.0

        try:
            feature_dim = (
                self.model.get_input_dim()
                if hasattr(self.model, 'get_input_dim')
                else int(getattr(self.model, 'input_dim', 7))
            )
            graph_items = []
            if components is None:
                components = list(nx.connected_components(graph))
            if len(components) > 1:
                collapse_target_nodes = self.collapse_target_ratio * self.original_nodes
                for nodes in components:
                    if not nodes or len(nodes) <= collapse_target_nodes:
                        continue
                    subgraph = graph.subgraph(nodes).copy()
                    graph_items.append((
                        self.to_pyg_data(subgraph, feature_dim=feature_dim),
                        subgraph.number_of_nodes(),
                    ))
                if not graph_items:
                    return 0.0
                if hasattr(self.model, 'predict_many'):
                    critical_values = self.model.predict_many(graph_items)
                else:
                    critical_values = [
                        self.model.predict(graph_data, current_nodes=size)
                        for graph_data, size in graph_items
                    ]
                collapse_distance = sum(
                    pred * size
                    for pred, (_graph_data, size) in zip(critical_values, graph_items)
                ) / self.original_nodes
                return float(collapse_distance)

            graph_data = self.to_pyg_data(graph, feature_dim=feature_dim)

            if hasattr(self.model, 'predict_collapse_distance'):
                return self.model.predict_collapse_distance(
                    graph_data,
                    current_nodes=current_nodes,
                    initial_nodes=self.original_nodes,
                )

            critical_threshold = self.model.predict(graph_data, current_nodes)
            return float(critical_threshold * (current_nodes / self.original_nodes))

        except Exception as e:
            self.model_error = str(e)
            print(f"Model prediction error: {e}")
            return np.nan

    @staticmethod
    def _ordered_nodes(G: nx.Graph) -> List:
        """Use deterministic node order where possible while supporting string IDs."""
        try:
            return sorted(G.nodes())
        except TypeError:
            return list(G.nodes())

    @staticmethod
    def _max_degree_node(G: nx.Graph):
        """Match the experiment degree-attack tie break when node IDs are comparable."""
        deg = dict(G.degree())
        if not deg:
            return None
        try:
            return max(deg, key=lambda node: (deg[node], node))
        except TypeError:
            return max(deg, key=lambda node: deg[node])

    @classmethod
    def calculate_node_features(cls, G: nx.Graph, feature_dim: int = 7) -> np.ndarray:
        """
        Compute node features in the same order used by process_network.py:
        degree, clustering, k-core, average-neighbor-degree, PageRank,
        betweenness centrality, eigenvector centrality.
        """
        nodes = cls._ordered_nodes(G)
        if not nodes:
            return np.empty((0, feature_dim), dtype=np.float32)
        if feature_dim <= 0:
            return np.empty((len(nodes), 0), dtype=np.float32)

        degree = dict(G.degree())
        columns = [[degree.get(node, 0.0) for node in nodes]]

        if feature_dim >= 2:
            clustering = nx.clustering(G)
            columns.append([clustering.get(node, 0.0) for node in nodes])

        if feature_dim >= 3:
            try:
                core_number = nx.core_number(G)
            except Exception:
                core_number = {node: 0.0 for node in nodes}
            columns.append([core_number.get(node, 0.0) for node in nodes])

        if feature_dim >= 4:
            try:
                avg_neighbor_degree = nx.average_neighbor_degree(G)
            except Exception:
                avg_neighbor_degree = {node: 0.0 for node in nodes}
            columns.append([avg_neighbor_degree.get(node, 0.0) for node in nodes])

        if feature_dim >= 5:
            try:
                pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
            except Exception:
                pagerank = {node: 1.0 / len(nodes) for node in nodes}
            columns.append([pagerank.get(node, 0.0) for node in nodes])

        if feature_dim >= 6:
            try:
                betweenness = nx.betweenness_centrality(G)
            except Exception:
                betweenness = {node: 0.0 for node in nodes}
            columns.append([betweenness.get(node, 0.0) for node in nodes])

        if feature_dim >= 7:
            try:
                eigenvector = nx.eigenvector_centrality(G, max_iter=500, tol=1e-5)
            except Exception:
                max_degree = max(degree.values()) if degree else 1.0
                eigenvector = {
                    node: degree.get(node, 0.0) / max(1.0, float(max_degree))
                    for node in nodes
                }
            columns.append([eigenvector.get(node, 0.0) for node in nodes])

        features = np.asarray(columns, dtype=np.float32).T

        if feature_dim > features.shape[1]:
            padding = np.zeros((features.shape[0], feature_dim - features.shape[1]), dtype=np.float32)
            features = np.hstack([features, padding])
        return features

    @classmethod
    def to_pyg_data(cls, G: nx.Graph, feature_dim: int = 7):
        """Convert the current NetworkX graph to PyG Data for TCR-GIN inference."""
        data_cls = _ensure_torch_geometric()

        nodes = cls._ordered_nodes(G)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        if G.number_of_edges() > 0:
            edges = np.array(
                [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()],
                dtype=np.int64,
            )
            row, col = edges[:, 0], edges[:, 1]
            edge_index = torch.from_numpy(
                np.array([np.concatenate([row, col]), np.concatenate([col, row])])
            ).long()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        x = torch.from_numpy(cls.calculate_node_features(G, feature_dim=feature_dim)).float()
        return data_cls(x=x, edge_index=edge_index)

    def attack_one_step(self) -> Optional[Dict]:
        """
        Execute one attack step.

        Returns:
            Dictionary with metrics, or None if attack sequence is exhausted
        """
        if self.step > len(self.attack_sequence):
            return None

        # Steps are 1-indexed, so the attack sequence index is step - 1.
        node = self.attack_sequence[self.step - 1]

        # Check if node exists
        if node not in self.current_G:
            warning = f"Node {node} is not in the network and was skipped."
            metrics = {
                'step': self.step,
                'node_removed': node,
                'warning': warning,
                'skipped': True
            }
        else:
            # Remove node
            self.current_G.remove_node(node)

            # Calculate metrics
            metrics = self.calculate_metrics()
            metrics['step'] = self.step
            metrics['node_removed'] = node
            metrics['skipped'] = False

        self.history.append(metrics)
        self.step += 1

        return metrics

    def attack_multiple_steps(self, num_steps: int) -> List[Dict]:
        """Execute multiple attack steps."""
        results = []
        for _ in range(num_steps):
            result = self.attack_one_step()
            if result is None:
                break
            results.append(result)
        return results

    def calculate_metrics(self) -> Dict:
        """
        Calculate current network metrics.

        Follow evaluate_classic_robustness.py semantics.
        """
        metrics = {}
        analysis_G = self.analysis_graph()

        # Basic info
        raw_current_nodes = self.current_G.number_of_nodes()
        raw_current_edges = self.current_G.number_of_edges()
        current_nodes = analysis_G.number_of_nodes()
        current_edges = analysis_G.number_of_edges()

        metrics['current_nodes'] = current_nodes
        metrics['current_edges'] = current_edges
        metrics['raw_current_nodes'] = raw_current_nodes
        metrics['raw_current_edges'] = raw_current_edges
        metrics['attacked_nodes'] = self.original_nodes - raw_current_nodes
        metrics['removed_nodes'] = self.original_nodes - current_nodes
        metrics['removed_ratio'] = metrics['removed_nodes'] / self.original_nodes

        # LCC (Largest Connected Component)
        components = []
        lcc_nodes = 0
        if current_nodes > 0:
            components = list(nx.connected_components(analysis_G))
            largest_cc = max(components, key=len)
            lcc_nodes = len(largest_cc)
            lcc_size = lcc_nodes / self.original_nodes

            metrics['lcc'] = lcc_nodes
            metrics['lcc_size'] = lcc_size
            metrics['num_components'] = len(components)
        else:
            metrics['lcc'] = 0
            metrics['lcc_size'] = 0.0
            metrics['num_components'] = 0

        # Collapse distance predicted by TCR-GIN.
        metrics['collapse_distance'] = self.predict_collapse_distance(
            graph=analysis_G,
            components=components,
            lcc_nodes=lcc_nodes,
        )

        # Natural connectivity
        metrics['natural_connectivity'] = (
            self.calculate_natural_connectivity(analysis_G)
            if self.compute_natural_connectivity
            else np.nan
        )

        # R-value: dynamic degree attack on the isolate-free residual graph.
        metrics['r_value'] = (
            self.calculate_r_value_dcr(analysis_G)
            if self.compute_r_value
            else np.nan
        )

        return metrics

    def set_collapse_target(self, collapse_target_ratio: float):
        """Set the LCC collapse threshold used to skip post-collapse prediction."""
        self.collapse_target_ratio = float(collapse_target_ratio or 0.0)

    def set_metric_options(
        self,
        compute_natural_connectivity: Optional[bool] = None,
        compute_r_value: Optional[bool] = None,
    ):
        """Update optional metric computation flags for future attack steps."""
        if compute_natural_connectivity is not None:
            self.compute_natural_connectivity = bool(compute_natural_connectivity)
        if compute_r_value is not None:
            self.compute_r_value = bool(compute_r_value)

    def calculate_natural_connectivity(self, graph: Optional[nx.Graph] = None) -> float:
        """
        Calculate natural connectivity.

        Natural connectivity = ln(sum(exp(λ_i))) - ln(N_initial)

        Reference: evaluate_classic_robustness.py line 430-438
        """
        graph = self.analysis_graph() if graph is None else graph
        if graph.number_of_nodes() == 0:
            return -10.0

        try:
            from scipy import linalg
            adj_array = nx.to_numpy_array(graph)
            eigenvalues = linalg.eigvalsh(adj_array)
            nat_conn = np.logaddexp.reduce(eigenvalues) - np.log(self.original_nodes)
            return float(nat_conn)
        except Exception as e:
            return 0.0

    def calculate_r_value_dcr(self, graph: Optional[nx.Graph] = None) -> float:
        """
        Calculate R(DCR) value - robustness under degree-based attack.

        Full logic follows evaluate_classic_robustness.py:
        1. Simulate a complete dynamic degree attack on the current remnant.
        2. Accumulate the current LCC size before each node removal.
        3. R = acc_sum / (N_remnant^2)

        Returns:
            R(DCR) value
        """
        graph = self.analysis_graph() if graph is None else graph
        if graph.number_of_nodes() == 0:
            return 0.0

        try:
            # Simulate on the same isolate-free remnant used by experiment files.
            g_sim = graph.copy()
            N_remnant = g_sim.number_of_nodes()
            acc_sum = 0.0

            for i in range(N_remnant):
                if g_sim.number_of_nodes() > 0:
                    if nx.is_connected(g_sim):
                        current_lcc_size = g_sim.number_of_nodes()
                    else:
                        components = list(nx.connected_components(g_sim))
                        if components:
                            current_lcc_size = len(max(components, key=len))
                        else:
                            current_lcc_size = 0
                else:
                    current_lcc_size = 0

                acc_sum += current_lcc_size

                node_to_remove = self._max_degree_node(g_sim)
                if node_to_remove is None:
                    break
                g_sim.remove_node(node_to_remove)

            # evaluate_classic_robustness.py normalizes by the current graph size.
            n_squared = N_remnant * N_remnant
            return acc_sum / n_squared if n_squared > 0 else 0.0

        except Exception as e:
            print(f"R-value calculation error: {e}")
            return 0.0

    def check_warning(self, warning_target_ratio: float) -> bool:
        """Check if warning target has been reached."""
        if not self.history:
            return False
        latest = self.history[-1]
        pred_dc = latest.get('collapse_distance')
        if pred_dc is not None and np.isfinite(pred_dc):
            return pred_dc <= warning_target_ratio
        return False

    def check_collapse(self, collapse_target_ratio: float) -> bool:
        """Check if network has collapsed."""
        if not self.history:
            return False
        latest = self.history[-1]
        if 'lcc_size' in latest:
            return latest['lcc_size'] <= collapse_target_ratio
        return False

    def get_metrics_dataframe(self):
        """Convert history to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.history)


class AttackSequenceGenerator:
    """Generate attack sequences using different strategies."""

    @staticmethod
    def random_sequence(G: nx.Graph, seed: Optional[int] = None) -> List:
        """Generate random attack sequence."""
        nodes = list(G.nodes())
        if seed is not None:
            np.random.seed(seed)
        return list(np.random.permutation(nodes))

    @staticmethod
    def degree_based_sequence(G: nx.Graph, descending: bool = True) -> List:
        """Generate degree-based attack sequence."""
        degrees = dict(G.degree())
        if descending:
            try:
                return sorted(degrees.keys(), key=lambda node: (-degrees[node], node))
            except TypeError:
                return sorted(degrees.keys(), key=lambda node: -degrees[node])
        try:
            return sorted(degrees.keys(), key=lambda node: (degrees[node], node))
        except TypeError:
            return sorted(degrees.keys(), key=lambda node: degrees[node])

    @staticmethod
    def betweenness_based_sequence(G: nx.Graph, descending: bool = True) -> List:
        """Generate betweenness-based attack sequence."""
        betweenness = nx.betweenness_centrality(G)
        sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=descending)
        return [node for node, bc in sorted_nodes]

    @staticmethod
    def load_from_csv(csv_content: str) -> List:
        """Load attack sequence from CSV string."""
        reader = csv.reader(StringIO(csv_content))
        sequence = []
        for row in reader:
            if row and row[0].strip():
                try:
                    node = int(row[0].strip())
                except ValueError:
                    node = row[0].strip()
                sequence.append(node)
        return sequence

    @staticmethod
    def map_sequence_to_internal_ids(sequence: List, node_mapping: Optional[Dict] = None) -> Tuple[List, Dict]:
        """Map a sequence containing original node IDs to internal 0..N-1 IDs."""
        if not node_mapping:
            return sequence, {"mapped": 0, "unchanged": len(sequence), "invalid": 0, "invalid_nodes": []}

        to_internal = node_mapping.get("to_internal", {})
        internal_nodes = set(node_mapping.get("to_original", {}).keys())
        mapped_sequence = []
        stats = {"mapped": 0, "unchanged": 0, "invalid": 0, "invalid_nodes": []}

        for node in sequence:
            candidates = [node]
            if isinstance(node, str):
                stripped = node.strip()
                candidates.append(stripped)
                try:
                    candidates.append(int(stripped))
                except ValueError:
                    pass
            elif isinstance(node, np.integer):
                candidates.append(int(node))

            mapped_node = None
            for candidate in candidates:
                if candidate in to_internal:
                    mapped_node = to_internal[candidate]
                    stats["mapped"] += 1
                    break
                if candidate in internal_nodes:
                    mapped_node = int(candidate)
                    stats["unchanged"] += 1
                    break

            if mapped_node is None:
                mapped_node = node
                stats["invalid"] += 1
                if len(stats["invalid_nodes"]) < 10:
                    stats["invalid_nodes"].append(node)
            mapped_sequence.append(mapped_node)

        return mapped_sequence, stats
