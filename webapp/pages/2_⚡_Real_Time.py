#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Monitoring page for attack simulation and early warning."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from webapp.core.validators import CollapseTargetValidator  # noqa: E402
from webapp.navigation import (  # noqa: E402
    init_session_state,
    render_appearance_selector,
    render_sidebar_nav,
)
from webapp.visualization.panels import (  # noqa: E402
    create_collapse_distance_panel,
    create_lcc_panel,
    create_natural_connectivity_panel,
    create_robustness_panel,
)


st.set_page_config(page_title="Monitoring", page_icon="⚡", layout="wide")
init_session_state()
render_sidebar_nav()
render_appearance_selector()

st.markdown(
    """
    <style>
        .control-note {
            color: var(--app-muted);
            font-size: 0.92rem;
            margin-top: -0.25rem;
        }
        .status-banner {
            border: 1px solid var(--app-border);
            border-left-width: 4px;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: var(--app-surface);
            color: var(--app-text);
            margin: 0.5rem 0 1rem 0;
        }
        .status-banner.warning {
            border-left-color: #ff6b6b;
            background: color-mix(in srgb, var(--app-surface) 82%, #ff4d4f 18%);
        }
        .status-banner.normal { border-left-color: #2dd4bf; }
        .status-banner.collapse {
            border-left-color: #ff3333;
            background: color-mix(in srgb, var(--app-surface) 78%, #b42318 22%);
        }
        @keyframes warningPulse {
            0%, 100% {
                box-shadow: 0 0 0 0 rgba(255, 51, 51, 0.38), 0 0 18px rgba(255, 51, 51, 0.22);
                border-color: rgba(255, 51, 51, 0.85);
            }
            50% {
                box-shadow: 0 0 0 8px rgba(255, 51, 51, 0.04), 0 0 30px rgba(255, 51, 51, 0.46);
                border-color: rgba(255, 120, 120, 1);
            }
        }
        @keyframes warningIconFlash {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.14); opacity: 0.72; }
        }
        .alarm-panel {
            display: flex;
            align-items: center;
            gap: 1rem;
            border: 1px solid rgba(255, 51, 51, 0.85);
            border-left: 6px solid #ff3333;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin: 0.75rem 0 1rem 0;
            color: #fff7f7;
            background:
                linear-gradient(135deg, rgba(180, 35, 24, 0.92), rgba(95, 10, 18, 0.86)),
                var(--app-surface);
            animation: warningPulse 1.1s ease-in-out infinite;
        }
        .alarm-icon {
            width: 46px;
            height: 46px;
            min-width: 46px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            color: #ffffff;
            background: #ff1f1f;
            font-size: 2rem;
            font-weight: 900;
            line-height: 1;
            animation: warningIconFlash 0.7s ease-in-out infinite;
        }
        .alarm-title {
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .alarm-body {
            color: #ffe7e7;
            font-size: 0.95rem;
            margin-top: 0.18rem;
        }
        .section-rule {
            height: 1px;
            background: linear-gradient(90deg, var(--app-accent), transparent);
            margin: 0.75rem 0 1.1rem 0;
        }
        div[data-testid="metric-container"] { min-height: 96px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def should_trigger_warning(result: dict, threshold: float) -> bool:
    pred_dc = result.get("collapse_distance")
    return pred_dc is not None and pd.notna(pred_dc) and pred_dc <= threshold


def handle_attack_result(result: dict, threshold: float) -> None:
    if result is None:
        return

    if not st.session_state.warning_triggered and should_trigger_warning(result, threshold):
        st.session_state.warning_triggered = True

    sim = st.session_state.simulator
    if sim is not None and not st.session_state.collapsed:
        if sim.check_collapse(st.session_state.collapse_target):
            st.session_state.collapsed = True
            st.session_state.is_auto_attacking = False


if not st.session_state.get("initialized", False):
    st.error("System setup is incomplete. Finish Network Setup before opening Monitoring.")
    st.stop()

for key, default in {
    "simulator": None,
    "attack_sequence": None,
    "is_auto_attacking": False,
    "warning_triggered": False,
    "collapsed": False,
    "show_lcc_panel": True,
    "show_distance_panel": True,
    "show_connectivity_panel": False,
    "show_robustness_panel": False,
    "compute_connectivity_realtime": False,
    "compute_robustness_realtime": False,
    "plot_window_steps": 300,
    "network_layout_cache": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

network = st.session_state.network
network_meta = st.session_state.network_metadata
num_nodes = int(network_meta["num_nodes"])

st.title("Monitoring")
st.caption("Simulate node-removal attacks and monitor early-warning signals for network structural collapse.")

setup_cols = st.columns([1.1, 1.1, 1.4])
with setup_cols[0]:
    st.metric("Network", f"{num_nodes} nodes", delta=f"{network_meta['num_edges']} edges")
with setup_cols[1]:
    model_info = st.session_state.model.get_info() if st.session_state.get("model") is not None else None
    if model_info:
        st.metric("Model Segments", model_info["num_segments"], delta=", ".join(model_info["segments"]))
    else:
        st.metric("Model Segments", "N/A")
with setup_cols[2]:
    st.caption("To run another analysis, return to Network Setup and load a new network or model.")

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.subheader("Control Console")

seq_col, target_col, run_col = st.columns([1.15, 1.0, 1.15])

with seq_col:
    st.markdown("#### 1. Attack Sequence")
    attack_mode = st.radio(
        "Attack Mode",
        ["Random Attack", "Degree-Based Attack", "Upload CSV"],
        horizontal=True,
        key="attack_mode",
    )

    if attack_mode == "Random Attack":
        seed = st.number_input("Random Seed", value=42)
        if st.button("Generate Random Sequence", use_container_width=True):
            from webapp.core.attack_simulator import AttackSequenceGenerator

            sequence = AttackSequenceGenerator.random_sequence(network, seed=seed)
            st.session_state.attack_sequence = sequence
            st.session_state.simulator = None
            st.success(f"Generated an attack sequence with {len(sequence)} nodes.")

    elif attack_mode == "Degree-Based Attack":
        order = st.selectbox("Attack Order", ["High to Low Degree", "Low to High Degree"])
        if st.button("Generate Degree Sequence", use_container_width=True):
            from webapp.core.attack_simulator import AttackSequenceGenerator

            sequence = AttackSequenceGenerator.degree_based_sequence(
                network,
                descending=(order == "High to Low Degree"),
            )
            st.session_state.attack_sequence = sequence
            st.session_state.simulator = None
            st.success(f"Generated an attack sequence with {len(sequence)} nodes.")

    else:
        uploaded_csv = st.file_uploader("Upload attack sequence CSV", type=["csv"])
        if uploaded_csv:
            from webapp.core.attack_simulator import AttackSequenceGenerator

            csv_content = uploaded_csv.getvalue().decode("utf-8")
            raw_sequence = AttackSequenceGenerator.load_from_csv(csv_content)
            node_mapping = (network_meta or {}).get("node_mapping", {})
            sequence, mapping_stats = AttackSequenceGenerator.map_sequence_to_internal_ids(
                raw_sequence,
                node_mapping,
            )
            st.session_state.attack_sequence = sequence
            st.session_state.simulator = None
            st.success(f"Loaded {len(sequence)} attack entries.")
            if mapping_stats["mapped"]:
                st.caption(
                    f"Mapped {mapping_stats['mapped']} CSV node ID(s) from original IDs to internal IDs."
                )
            if mapping_stats["invalid"]:
                st.warning(
                    f"{mapping_stats['invalid']} CSV node ID(s) were not found in the network and will be skipped. "
                    f"Examples: {mapping_stats['invalid_nodes']}"
                )

    if st.session_state.attack_sequence:
        st.caption(f"Current sequence length: {len(st.session_state.attack_sequence)}")
    else:
        st.caption("No attack sequence has been generated.")

with target_col:
    st.markdown("#### 2. Warning Target")
    warning_target = st.number_input(
        "Warning target (integer steps/nodes)",
        min_value=1,
        max_value=num_nodes,
        value=3,
        step=1,
        help="For single-node attacks, this integer is equivalent to the number of attack steps. The app uses target / N0.",
    )
    decision_warning_threshold, warning_desc = CollapseTargetValidator.parse_warning_target(
        warning_target,
        num_nodes,
    )
    st.metric("Decision Threshold", f"{decision_warning_threshold:.4f}", delta=warning_desc)
    st.markdown(
        "<div class='control-note'>Warning is fixed to Collapse Distance: "
        "trigger when Collapse Distance <= target / N0.</div>",
        unsafe_allow_html=True,
    )

with run_col:
    st.markdown("#### 3. Run")
    attack_execution_mode = st.radio(
        "Execution Mode",
        ["Manual", "Automatic"],
        horizontal=True,
        key="attack_execution_mode",
    )
    if attack_execution_mode == "Automatic":
        speed_option = st.select_slider(
            "Attack Interval",
            options=["100 ms", "1 s", "10 s"],
            value="1 s",
        )
        speed_ms = {"100 ms": 100, "1 s": 1000, "10 s": 10000}[speed_option]
    else:
        speed_ms = 0

    show_network_view = st.checkbox(
        "Show Network View",
        value=False,
        help="Topology rendering can be slow. The initial layout is cached and reused once enabled.",
    )
    compute_connectivity_realtime = st.checkbox(
        "Compute Natural Connectivity",
        value=st.session_state.compute_connectivity_realtime,
        help="Requires eigenvalue decomposition and can slow down large networks.",
    )
    compute_robustness_realtime = st.checkbox(
        "Compute R(DCR)",
        value=st.session_state.compute_robustness_realtime,
        help="Runs a full dynamic degree attack at each step and is the most expensive optional metric.",
    )
    plot_window_steps = st.number_input(
        "Recent Steps Displayed",
        min_value=50,
        max_value=5000,
        value=int(st.session_state.plot_window_steps),
        step=50,
    )
    st.session_state.compute_connectivity_realtime = compute_connectivity_realtime
    st.session_state.compute_robustness_realtime = compute_robustness_realtime
    st.session_state.plot_window_steps = int(plot_window_steps)

    init_disabled = not st.session_state.attack_sequence
    if st.button("Initialize Simulator", type="primary", use_container_width=True, disabled=init_disabled):
        from webapp.core.attack_simulator import AttackSimulator

        model = st.session_state.get("model", None)
        st.session_state.simulator = AttackSimulator(
            network,
            st.session_state.attack_sequence,
            model=model,
            collapse_target_ratio=st.session_state.collapse_target,
            compute_natural_connectivity=compute_connectivity_realtime,
            compute_r_value=compute_robustness_realtime,
        )
        st.session_state.warning_triggered = False
        st.session_state.collapsed = False
        st.session_state.is_auto_attacking = False

action_cols = st.columns([1, 1, 1, 2])
with action_cols[0]:
    if st.button(
        "Attack One Step",
        use_container_width=True,
        disabled=st.session_state.simulator is None or attack_execution_mode != "Manual",
    ):
        result = st.session_state.simulator.attack_one_step()
        if result:
            handle_attack_result(result, decision_warning_threshold)
        else:
            st.info("The attack sequence has finished.")

with action_cols[1]:
    if attack_execution_mode == "Automatic":
        if st.button("Start Auto", use_container_width=True, disabled=st.session_state.simulator is None):
            st.session_state.is_auto_attacking = True
    else:
        st.button("Start Auto", use_container_width=True, disabled=True)

with action_cols[2]:
    if st.button("Stop", use_container_width=True, disabled=not st.session_state.get("is_auto_attacking", False)):
        st.session_state.is_auto_attacking = False

with action_cols[3]:
    if st.button("Reset Simulator", use_container_width=True, disabled=st.session_state.simulator is None):
        st.session_state.simulator.reset()
        st.session_state.simulator.set_collapse_target(st.session_state.collapse_target)
        st.session_state.warning_triggered = False
        st.session_state.collapsed = False
        st.session_state.is_auto_attacking = False

st.markdown("---")

if st.session_state.simulator is None:
    st.info("Choose an attack sequence and initialize the simulator.")
    st.stop()

sim = st.session_state.simulator
latest = sim.history[-1] if sim.history else None
plot_history = sim.history[-int(st.session_state.plot_window_steps):]

status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    completed_steps = max(0, sim.step - 1)
    st.metric("Attack Progress", f"{completed_steps}/{len(sim.attack_sequence)}")

with status_col2:
    if latest:
        lcc_nodes = latest.get("lcc", 0)
        lcc_size = latest.get("lcc_size", 0)
        st.metric("Current LCC", f"{lcc_nodes} nodes", delta=f"Size: {lcc_size:.2%}")
    else:
        st.metric("Current LCC", "Not started")

with status_col3:
    if latest:
        pred_dc = latest.get("collapse_distance")
        if pred_dc is not None and pd.notna(pred_dc):
            collapse_distance_nodes = max(0, int(round(float(pred_dc) * sim.original_nodes)))
            st.metric(
                "Distance to Collapse",
                f"{collapse_distance_nodes} nodes",
                delta=f"Collapse Distance: {pred_dc:.4f}",
            )
        else:
            st.metric("Distance to Collapse", "N/A", delta="Collapse Distance: N/A")
    else:
        st.metric("Distance to Collapse", "Not started")

with status_col4:
    if st.session_state.collapsed:
        st.error("Collapsed")
    elif st.session_state.warning_triggered:
        st.warning("Warning")
    else:
        st.success("Normal")

if getattr(sim, "model_error", None):
    st.warning(f"Model prediction is unavailable; Collapse Distance was recorded as NaN: {sim.model_error}")

if latest:
    pred_dc = latest.get("collapse_distance")
    pred_text = f"{pred_dc:.4f}" if pred_dc is not None and pd.notna(pred_dc) else "N/A"
    if st.session_state.collapsed:
        st.markdown(
            "<div class='status-banner collapse'><b>Network structure collapsed.</b> "
            "The largest connected component reached the Collapse Target.</div>",
            unsafe_allow_html=True,
        )
    elif st.session_state.warning_triggered:
        st.markdown(
            f"""
            <div class='alarm-panel'>
                <div class='alarm-icon'>!</div>
                <div>
                    <div class='alarm-title'>Structural Collapse Warning</div>
                    <div class='alarm-body'>
                        Collapse Distance = {pred_text}; Warning Target = {decision_warning_threshold:.4f}.
                        Immediate inspection is recommended.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='status-banner warning'><b>Warning reached.</b> "
            f"Collapse Distance = {pred_text}; Warning Target = {decision_warning_threshold:.4f}.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='status-banner normal'><b>Warning Target:</b> "
            f"{int(warning_target)} step(s) / {num_nodes} nodes = {decision_warning_threshold:.4f}; "
            f"latest Collapse Distance = {pred_text}.</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.subheader("Monitoring Panels")
col1, col2, col3, col4 = st.columns(4)
with col1:
    show_lcc = st.checkbox("LCC Size", key="show_lcc_panel")
with col2:
    show_distance = st.checkbox("Collapse Distance", key="show_distance_panel")
with col3:
    show_connectivity = st.checkbox("Natural Connectivity", key="show_connectivity_panel")
with col4:
    show_robustness = st.checkbox("R(DCR)", key="show_robustness_panel")

if sim is not None:
    sim.set_metric_options(
        compute_natural_connectivity=show_connectivity or compute_connectivity_realtime,
        compute_r_value=show_robustness or compute_robustness_realtime,
    )

if show_connectivity and not any(np.isfinite(h.get("natural_connectivity", np.nan)) for h in sim.history):
    st.caption("Natural Connectivity is enabled and will be computed from the next attack step.")
if show_robustness and not any(np.isfinite(h.get("r_value", np.nan)) for h in sim.history):
    st.caption("R(DCR) is enabled and will be computed from the next attack step.")
if len(sim.history) > len(plot_history):
    st.caption(
        f"For responsiveness, charts show the latest {len(plot_history)} steps. "
        f"CSV export still includes all {len(sim.history)} records."
    )

selected_panels = sum([show_lcc, show_distance, show_connectivity, show_robustness])
st.markdown("---")

if selected_panels == 0:
    st.info("Select at least one monitoring panel.")
elif selected_panels == 1:
    if show_lcc:
        st.plotly_chart(
            create_lcc_panel(plot_history, st.session_state.collapse_target, None, num_nodes),
            use_container_width=True,
            key="lcc_single",
        )
    elif show_distance:
        st.plotly_chart(
            create_collapse_distance_panel(
                plot_history,
                st.session_state.collapse_target,
                num_nodes,
                sim.initial_collapse_distance,
                decision_warning_threshold,
            ),
            use_container_width=True,
            key="dist_single",
        )
    elif show_connectivity:
        st.plotly_chart(create_natural_connectivity_panel(plot_history), use_container_width=True, key="nat_single")
    elif show_robustness:
        st.plotly_chart(create_robustness_panel(plot_history), use_container_width=True, key="rob_single")
elif selected_panels == 2:
    panel_cols = st.columns(2)
    panels = []
    if show_lcc:
        panels.append(("lcc", create_lcc_panel(plot_history, st.session_state.collapse_target, None, num_nodes)))
    if show_distance:
        panels.append((
            "distance",
            create_collapse_distance_panel(
                plot_history,
                st.session_state.collapse_target,
                num_nodes,
                sim.initial_collapse_distance,
                decision_warning_threshold,
            ),
        ))
    if show_connectivity:
        panels.append(("connectivity", create_natural_connectivity_panel(plot_history)))
    if show_robustness:
        panels.append(("robustness", create_robustness_panel(plot_history)))
    for idx, (panel_name, panel_fig) in enumerate(panels):
        with panel_cols[idx]:
            st.plotly_chart(panel_fig, use_container_width=True, key=f"{panel_name}_col{idx}")
else:
    panel_cols = st.columns(2)
    panels = []
    if show_lcc:
        panels.append(create_lcc_panel(plot_history, st.session_state.collapse_target, None, num_nodes))
    if show_distance:
        panels.append(create_collapse_distance_panel(
            plot_history,
            st.session_state.collapse_target,
            num_nodes,
            sim.initial_collapse_distance,
            decision_warning_threshold,
        ))
    if show_connectivity:
        panels.append(create_natural_connectivity_panel(plot_history))
    if show_robustness:
        panels.append(create_robustness_panel(plot_history))
    for idx, panel in enumerate(panels):
        with panel_cols[idx % 2]:
            st.plotly_chart(panel, use_container_width=True, key=f"panel_{idx}_{selected_panels}")

if show_network_view:
    st.markdown("---")
    st.subheader("Network State")
    try:
        current_G = sim.current_G
        if current_G.number_of_nodes() > 0:
            latest = sim.history[-1]
            lcc_size = latest.get("lcc_size", 0)
            from webapp.visualization.network_viz import (
                compute_network_layout,
                create_network_visualization,
                layout_axis_range,
            )

            layout_key = (id(network), network.number_of_nodes(), network.number_of_edges())
            layout_cache = st.session_state.network_layout_cache
            if layout_key not in layout_cache:
                layout_pos = compute_network_layout(network)
                layout_cache.clear()
                layout_cache[layout_key] = {
                    "pos": layout_pos,
                    "axis_range": layout_axis_range(layout_pos),
                }
            fixed_layout = layout_cache[layout_key]

            fig = create_network_visualization(
                current_G,
                title=f"Current network - {current_G.number_of_nodes()} nodes, LCC={lcc_size:.3f}",
                pos=fixed_layout["pos"],
                axis_range=fixed_layout["axis_range"],
            )
            st.plotly_chart(fig, use_container_width=True, key=f"network_viz_{sim.step}")
        else:
            st.warning("No nodes remain in the network.")
    except Exception as exc:
        st.error(f"Network visualization failed: {exc}")

if attack_execution_mode == "Automatic" and st.session_state.get("is_auto_attacking", False):
    if sim.step <= len(sim.attack_sequence):
        result = sim.attack_one_step()
        if result is not None:
            handle_attack_result(result, decision_warning_threshold)
        time.sleep(speed_ms / 1000)
        st.rerun()
    else:
        st.session_state.is_auto_attacking = False
        st.success("The attack sequence has finished.")

st.markdown("---")
if sim.history:
    export_col, table_col, _ = st.columns([1, 1, 2])
    df = sim.get_metrics_dataframe()
    with export_col:
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Results (CSV)",
            csv,
            "attack_results.csv",
            "text/csv",
            use_container_width=True,
        )
    with table_col:
        if st.button("Show Data Table", use_container_width=True):
            st.dataframe(df, use_container_width=True)
