#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User guide page for the TCR-GIN web application."""

import streamlit as st

from webapp.navigation import init_session_state, render_appearance_selector, render_sidebar_nav


st.set_page_config(page_title="User Guide", page_icon="Guide", layout="wide")
init_session_state()
render_sidebar_nav()
render_appearance_selector()

st.title("User Guide")

st.header("Overview")
st.markdown(
    """
    TCR-GIN monitors a network under node-removal attacks and reports early
    warning signals from four panels: LCC Size, Collapse Distance, Natural
    Connectivity, and R(DCR).

    The real-time warning rule follows the experiment scripts: the warning
    threshold is entered as an integer target `k`, then converted to `k / N0`,
    where `N0` is the initial network size. During monitoring, the alert is
    triggered when `Collapse Distance <= k / N0`.

    Collapse Distance follows the experiment logic. If the current residual
    graph has multiple connected components, components larger than the collapse
    target are predicted separately and aggregated by node count. The final
    Collapse Distance is normalized to the initial network scale, so the
    node-count distance shown in the status area is `Collapse Distance * N0`.
    """
)

st.header("Network Setup")
st.markdown(
    """
    1. Upload a `.npz` network file.
    2. The file may contain `edges`, `data`, or `edge_index`.
    3. Supported edge shapes are `E x 2` and PyG-style `2 x E`.
    4. Set the collapse target as either a ratio, such as `0.3`, or an absolute
       LCC node count, such as `150`.
    5. Load the TCR-GIN model from a local directory or a ZIP archive, then
       upload the matching YAML configuration.
    """
)

st.code(
    """import numpy as np

edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
])
np.savez("network_edges.npz", edges=edges)
""",
    language="python",
)

st.header("Model Loading")
st.markdown(
    """
    The model loader uses the YAML configuration to map model folders to node
    ranges. Folder names do not need to be pure ranges. For example,
    `100-200-transport` can be mapped to `[0, 300]` when the configuration
    contains:
    """
)
st.code(
    """model_suite:
  - name: "Base-Model-0-300"
    node_range: [0, 300]
    base_dir: "../models/transport-models/100-200-transport"

  - name: "Base-Model-300-400"
    node_range: [300, 400]
    base_dir: "../models/transport-models/300-transport"
""",
    language="yaml",
)
st.markdown(
    """
    If multiple `model_run_*.pt` files are present in the same segment, the app
    selects `model_run_1.pt`, matching the experiment scripts.
    """
)

st.header("Real-Time Monitoring")
st.markdown(
    """
    1. Choose an attack sequence: random, degree-based, or CSV upload.
    2. Set **Warning Target** as an integer. In single-node attacks, this is
       equivalent to the number of attack steps. The app displays the converted
       ratio `target / N0`.
    3. Initialize the simulator.
    4. Run one attack step manually or use automatic mode.

    The Collapse Distance panel marks the decision threshold and highlights the
    warning point once the warning condition is reached.
    """
)

st.header("Output")
st.markdown(
    """
    After running an attack, download the CSV file to inspect per-step metrics:
    removed node, LCC, Collapse Distance, natural connectivity, and R(DCR).
    """
)
