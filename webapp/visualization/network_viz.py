#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Network Visualization Module - Fixed Version

Creates Gephi-style interactive network visualizations.
"""

import networkx as nx
import plotly.graph_objects as go
from typing import Optional, List, Dict
import numpy as np

try:
    import streamlit as st
except Exception:
    st = None


def _network_style() -> dict:
    if st is not None and st.session_state.get("appearance_mode") == "Dark":
        return {
            "bg": "#171d24",
            "text": "#f3f6fa",
            "edge": "rgba(181, 192, 207, 0.32)",
            "line": "#171d24",
        }
    return {
        "bg": "#ffffff",
        "text": "#172033",
        "edge": "rgba(91, 104, 124, 0.35)",
        "line": "#ffffff",
    }


def compute_network_layout(G: nx.Graph, layout_algorithm: str = 'spring') -> Dict:
    """Compute a reusable node layout for network snapshots."""
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_nodes() > 500:
        return nx.spring_layout(G, k=0.5, iterations=20, seed=42)
    if layout_algorithm == 'kamada_kawai':
        return nx.kamada_kawai_layout(G)
    if layout_algorithm == 'circular':
        return nx.circular_layout(G)
    return nx.spring_layout(G, k=1 / np.sqrt(G.number_of_nodes()), iterations=50, seed=42)


def layout_axis_range(pos: Dict, padding: float = 0.08) -> Optional[Dict[str, List[float]]]:
    """Return fixed x/y ranges for a layout dictionary."""
    if not pos:
        return None
    coords = np.asarray(list(pos.values()), dtype=float)
    if coords.size == 0:
        return None
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)
    return {
        "x": [float(x_min - x_span * padding), float(x_max + x_span * padding)],
        "y": [float(y_min - y_span * padding), float(y_max + y_span * padding)],
    }


def create_network_visualization(
    G: nx.Graph,
    highlighted_nodes: Optional[List] = None,
    node_colors: Optional[Dict] = None,
    layout_algorithm: str = 'spring',
    title: str = 'Network Visualization',
    show_labels: bool = False,
    pos: Optional[Dict] = None,
    axis_range: Optional[Dict[str, List[float]]] = None,
) -> go.Figure:
    """
    Create interactive network visualization with Gephi-style aesthetics.

    Args:
        G: NetworkX graph
        highlighted_nodes: List of nodes to highlight
        node_colors: Dictionary mapping nodes to color values
        layout_algorithm: 'spring', 'kamada_kawai', or 'circular'
        title: Plot title
        show_labels: Whether to show node labels
        pos: Optional precomputed node positions. Nodes absent from the current
            graph are simply not drawn.
        axis_range: Optional fixed x/y axis ranges from the original layout.

    Returns:
        Plotly Figure object
    """
    style = _network_style()

    if G.number_of_nodes() == 0:
        # Empty graph
        fig = go.Figure()
        fig.update_layout(
            title="Network Empty",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor=style["bg"],
            paper_bgcolor=style["bg"],
            font=dict(color=style["text"]),
        )
        return fig

    # Calculate layout only when a fixed layout was not supplied.
    if pos is None:
        pos = compute_network_layout(G, layout_algorithm)
    if axis_range is None:
        axis_range = layout_axis_range(pos)

    # Create edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        if edge[0] not in pos or edge[1] not in pos:
            continue
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.6, color=style["edge"]),
        hoverinfo='none',
        mode='lines',
        opacity=1.0
    )

    # Create nodes
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []

    visible_nodes = [node for node in G.nodes() if node in pos]
    for node in visible_nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        degree = G.degree(node)
        node_text.append(f'Node: {node}<br>Degree: {degree}')

        # Node size based on degree
        size = 8 + degree * 1.5
        node_size.append(min(size, 30))  # Cap at 30

        # Node color
        if node_colors and node in node_colors:
            node_color.append(node_colors[node])
        elif highlighted_nodes and node in highlighted_nodes:
            node_color.append('red')
        else:
            node_color.append(degree)

    # Create node trace without colorbar to avoid compatibility issues
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if show_labels else 'markers',
        hoverinfo='text',
        text=[str(node) for node in visible_nodes] if show_labels else None,
        hovertext=node_text,
        textposition="top center",
        textfont=dict(size=8, color=style["text"]),
        marker=dict(
            showscale=False,  # Disable colorbar
            colorscale='Bluered',
            size=node_size,
            color=node_color,
            opacity=0.92,
            line=dict(width=1, color=style["line"])
        )
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(color=style["text"], size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=axis_range.get("x") if axis_range else None,
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=axis_range.get("y") if axis_range else None,
            ),
            plot_bgcolor=style["bg"],
            paper_bgcolor=style["bg"],
            font=dict(color=style["text"]),
            height=600
        )
    )

    return fig
