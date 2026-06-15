#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitoring Panels Module

Creates real-time monitoring panels for network metrics.
"""

import plotly.graph_objects as go
from typing import List, Dict, Optional
import numpy as np

try:
    import streamlit as st
except Exception:
    st = None


LIGHT_STYLE = {
    "template": "plotly_white",
    "font": "#172033",
    "grid": "rgba(23, 32, 51, 0.10)",
    "plot_bg": "#ffffff",
    "paper_bg": "#ffffff",
    "label_bg": "#ffffff",
}
DARK_STYLE = {
    "template": "plotly_dark",
    "font": "#f3f6fa",
    "grid": "rgba(243, 246, 250, 0.14)",
    "plot_bg": "#171d24",
    "paper_bg": "#171d24",
    "label_bg": "#111820",
}


def _panel_style() -> dict:
    if st is not None and st.session_state.get("appearance_mode") == "Dark":
        return DARK_STYLE
    return LIGHT_STYLE


def _base_layout(height: int = 350) -> dict:
    style = _panel_style()
    return dict(
        template=style["template"],
        height=height,
        plot_bgcolor=style["plot_bg"],
        paper_bgcolor=style["paper_bg"],
        font=dict(color=style["font"], size=12),
        title_font=dict(size=16, color=style["font"]),
        hovermode='x unified',
        margin=dict(l=60, r=24, t=46, b=44),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )


def _axis_style(extra: Optional[dict] = None) -> dict:
    style_values = _panel_style()
    axis_line = "rgba(243, 246, 250, 0.25)" if style_values is DARK_STYLE else "rgba(23, 32, 51, 0.18)"
    tick_color = "rgba(243, 246, 250, 0.42)" if style_values is DARK_STYLE else "rgba(23, 32, 51, 0.25)"
    style = dict(
        showgrid=True,
        gridcolor=style_values["grid"],
        zeroline=False,
        linecolor=axis_line,
        tickcolor=tick_color,
    )
    if extra:
        style.update(extra)
    return style


def _threshold_annotation(fig: go.Figure, y: float, text: str, color: str) -> None:
    style = _panel_style()
    fig.add_annotation(
        x=0.012,
        y=y,
        xref="paper",
        yref="y",
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        yshift=-5,
        font=dict(size=11, color=color),
        bgcolor=style["label_bg"],
        bordercolor=color,
        borderwidth=1,
        borderpad=3,
    )


def _step_dtick(steps: List[int]) -> int:
    if not steps:
        return 1
    max_step = max(steps)
    if max_step <= 50:
        return 1
    if max_step <= 100:
        return 5
    if max_step <= 500:
        return 10
    return 20


def create_lcc_panel(
    history: List[Dict],
    collapse_target_ratio: float,
    warning_target_ratio: Optional[float] = None,
    original_nodes: int = 0
) -> go.Figure:
    """
    Create LCC Size monitoring panel.

    LCC Size is the largest connected component size normalized by N0.

    Args:
        history: List of metric dictionaries from attack simulator
        collapse_target_ratio: Collapse threshold (0-1)
        warning_target_ratio: Warning threshold (0-1), optional
        original_nodes: Original number of nodes

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    if not history:
        # Empty panel with axes
        fig.update_layout(
            title="📊 LCC Size",
            xaxis_title="Attack Step",
            yaxis_title="LCC Size",
            xaxis=_axis_style(),
            yaxis=_axis_style(dict(range=[0, 1.05])),
            **_base_layout()
        )
        return fig

    steps = [h['step'] for h in history if 'lcc_size' in h]
    lcc_sizes = [h['lcc_size'] for h in history if 'lcc_size' in h]

    fig.add_trace(go.Scatter(
        x=steps,
        y=lcc_sizes,
        mode='lines',
        name='LCC Size',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='Attack Step: %{x}<br>LCC Size: %{y:.3f}<extra></extra>'
    ))

    fig.add_hline(
        y=collapse_target_ratio,
        line_dash="dash",
        line_color="rgba(31, 119, 180, 0.8)",
        line_width=2
    )
    _threshold_annotation(
        fig,
        collapse_target_ratio,
        f"Collapse Target ({collapse_target_ratio:.2f})",
        "#6ea8fe" if _panel_style() is DARK_STYLE else "#1f77b4",
    )

    if warning_target_ratio is not None:
        fig.add_hline(
            y=warning_target_ratio,
            line_dash="dot",
            line_color="rgba(255, 127, 14, 0.85)",
            line_width=2
        )
        _threshold_annotation(
            fig,
            warning_target_ratio,
            f"Warning Target ({warning_target_ratio:.2f})",
            "#ffb86b",
        )

    fig.update_layout(
        title="📊 LCC Size",
        xaxis_title="Attack Step",
        yaxis_title="LCC Size",
        xaxis=_axis_style(dict(
            dtick=_step_dtick(steps),
            tick0=0,
        )),
        yaxis=_axis_style(dict(
            range=[0, 1.05],
            tickformat='.2f'
        )),
        **_base_layout()
    )

    return fig


def create_collapse_distance_panel(
    history: List[Dict],
    collapse_target_ratio: float,
    original_nodes: int,
    initial_collapse_distance: Optional[float] = None,
    warning_threshold: Optional[float] = None
) -> go.Figure:
    """
    Create collapse distance monitoring panel.

    Collapse Distance predicted by TCR-GIN and normalized to N0.

    Args:
        history: List of metric dictionaries
        collapse_target_ratio: Collapse threshold (unused here)
        original_nodes: Original number of nodes (unused here)
        initial_collapse_distance: Initial predicted collapse distance.

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    valid_distances = [
        h.get('collapse_distance')
        for h in history
        if h.get('collapse_distance') is not None and np.isfinite(h.get('collapse_distance'))
    ]
    if initial_collapse_distance is not None and not np.isfinite(initial_collapse_distance):
        initial_collapse_distance = None

    if initial_collapse_distance is None:
        initial_collapse_distance = valid_distances[0] if valid_distances else None
    if warning_threshold is None and original_nodes:
        warning_threshold = 3.0 / original_nodes

    max_distance = max(valid_distances) if valid_distances else (initial_collapse_distance or 1.0)
    if warning_threshold is not None:
        max_distance = max(max_distance, warning_threshold)
    max_y = max(0.01, max_distance * 1.2)

    lower_y = 0

    if not history or not valid_distances:
        # Empty panel with axes
        fig.add_annotation(
            text="Waiting for model prediction..." if history else "Waiting for data...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=18, color="gray")
        )
        fig.update_layout(
            title="📉 Collapse Distance",
            xaxis_title="Attack Step",
            yaxis_title="Collapse Distance",
            xaxis=_axis_style(),
            yaxis=_axis_style(dict(range=[lower_y, max_y], tickformat='.4f')),
            **_base_layout()
        )
        return fig

    points = [
        (h['step'], h['collapse_distance'])
        for h in history
        if 'collapse_distance' in h and np.isfinite(h['collapse_distance'])
    ]
    steps = [p[0] for p in points]
    distances = [p[1] for p in points]

    fig.add_trace(go.Scatter(
        x=steps,
        y=distances,
        mode='lines',
        name='Collapse Distance',
        line=dict(color='#ff0000', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)',
        hovertemplate='Attack Step: %{x}<br>Collapse Distance: %{y:.4f}<extra></extra>'
    ))

    if warning_threshold is not None:
        fig.add_hline(
            y=warning_threshold,
            line_dash="dot",
            line_color="rgba(255, 0, 0, 0.8)",
            line_width=2
        )
        _threshold_annotation(
            fig,
            warning_threshold,
            f"Warning Target ({warning_threshold:.4f})",
            "#ff6b6b",
        )

    warning_triggered = (
        warning_threshold is not None
        and any(distance <= warning_threshold for distance in distances)
    )

    if warning_triggered and steps:
        warning_idx = len(steps) - 1
        if warning_threshold is not None:
            for idx, distance in enumerate(distances):
                if distance <= warning_threshold:
                    warning_idx = idx
                    break
        warning_step = steps[warning_idx]
        warning_distance = distances[warning_idx]
        fig.add_trace(go.Scatter(
            x=[warning_step],
            y=[warning_distance],
            mode='markers+text',
            name='Warning triggered',
            marker=dict(color='#ff3333', size=13, symbol='diamond'),
            text=['WARNING'],
            textposition='top center',
            textfont=dict(color='#ff3333', size=12),
            hovertemplate='Warning triggered<br>Step: %{x}<br>Collapse Distance: %{y:.4f}<extra></extra>'
        ))
        fig.add_vline(
            x=warning_step,
            line_dash="dash",
            line_color="rgba(255, 51, 51, 0.75)",
            line_width=2
        )

    fig.update_layout(
        title="📉 Collapse Distance",
        xaxis_title="Attack Step",
        yaxis_title="Collapse Distance",
        xaxis=_axis_style(dict(dtick=_step_dtick(steps))),
        yaxis=_axis_style(dict(range=[lower_y, max_y], tickformat='.4f')),
        **_base_layout()
    )

    return fig


def create_natural_connectivity_panel(history: List[Dict]) -> go.Figure:
    """Create natural connectivity monitoring panel."""
    fig = go.Figure()

    if not history:
        fig.update_layout(
            title="🔗 Natural Connectivity",
            xaxis_title="Attack Step",
            yaxis_title="Natural Connectivity",
            xaxis=_axis_style(),
            yaxis=_axis_style(),
            **_base_layout()
        )
        return fig

    steps = [h['step'] for h in history if 'natural_connectivity' in h]
    nat_conn = [h['natural_connectivity'] for h in history if 'natural_connectivity' in h]

    fig.add_trace(go.Scatter(
        x=steps,
        y=nat_conn,
        mode='lines',
        name='Natural Connectivity',
        line=dict(color='#2ca02c', width=2.5),
        hovertemplate='Attack Step: %{x}<br>Natural Connectivity: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title="🔗 Natural Connectivity",
        xaxis_title="Attack Step",
        yaxis_title="Natural Connectivity",
        xaxis=_axis_style(dict(dtick=_step_dtick(steps))),
        yaxis=_axis_style(),
        **_base_layout()
    )

    return fig


def create_robustness_panel(history: List[Dict]) -> go.Figure:
    """Create R-value (robustness) monitoring panel."""
    fig = go.Figure()

    if not history:
        fig.update_layout(
            title="💪 R(DCR)",
            xaxis_title="Attack Step",
            yaxis_title="R-value",
            xaxis=_axis_style(),
            yaxis=_axis_style(dict(autorange=True)),
            **_base_layout()
        )
        return fig

    steps = [h['step'] for h in history if 'r_value' in h]
    r_values = [h['r_value'] for h in history if 'r_value' in h]

    fig.add_trace(go.Scatter(
        x=steps,
        y=r_values,
        mode='lines',
        name='R(DCR)',
        line=dict(color='#9467bd', width=2.5),
        hovertemplate='Attack Step: %{x}<br>R-value: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title="💪 R(DCR)",
        xaxis_title="Attack Step",
        yaxis_title="R-value",
        xaxis=_axis_style(dict(dtick=_step_dtick(steps))),
        yaxis=_axis_style(dict(autorange=True)),
        **_base_layout()
    )

    return fig


def _create_empty_panel(title: str) -> go.Figure:
    """Create an empty panel placeholder."""
    fig = go.Figure()
    fig.add_annotation(
        text="Waiting for data...",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="gray")
    )
    fig.update_layout(
        title=title,
        **_base_layout(),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig
