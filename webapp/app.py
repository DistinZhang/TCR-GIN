#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main Streamlit entry point for the network-collapse warning app."""

import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from webapp.navigation import (  # noqa: E402
    init_session_state,
    render_appearance_selector,
    render_home_links,
    render_sidebar_nav,
)


APP_NAME = "Step-by-step: early warning of network breakdown with collapse distance"
HOME_CONCEPT_IMAGE = Path(__file__).resolve().parent / "assets" / "home_concept.png"


st.set_page_config(
    page_title=APP_NAME,
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_sidebar_nav()
render_appearance_selector()

st.markdown(
    """
    <style>
        .main { padding: 0rem 1rem; }
        .stButton > button { width: 100%; }
        .hero-band {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            background: var(--app-surface);
            color: var(--app-text);
            box-shadow: 0 1px 2px var(--app-shadow);
            margin-bottom: 1rem;
        }
        .home-concept {
            margin: 0.75rem 0 1.25rem 0;
        }
        .home-concept img {
            border-radius: 8px;
            border: 1px solid var(--app-border);
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
        }
        .home-concept-caption {
            margin-top: 0.45rem;
            color: var(--app-muted);
            font-size: 0.82rem;
            line-height: 1.35;
        }
        .metric-card {
            padding: 1rem;
            border-radius: 8px;
            background-color: var(--app-surface);
            border: 1px solid var(--app-border);
            color: var(--app-text);
            box-shadow: 0 1px 2px var(--app-shadow);
        }
        .status-ready {
            border-left: 4px solid #2f855a;
            background-color: color-mix(in srgb, var(--app-surface) 86%, #2f855a 14%);
        }
        .status-pending {
            border-left: 4px solid #d99000;
            background-color: color-mix(in srgb, var(--app-surface) 84%, #d99000 16%);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_NAME)

st.markdown(
    """
    <div class="hero-band">
    Early warning for network structural collapse under node-removal attacks, powered by TCR-GIN.
    </div>
    """,
    unsafe_allow_html=True,
)

if HOME_CONCEPT_IMAGE.exists():
    st.markdown('<div class="home-concept">', unsafe_allow_html=True)
    st.image(str(HOME_CONCEPT_IMAGE), width="stretch")
    st.markdown(
        """
        <div class="home-concept-caption">
        Conceptual illustration of the early-warning workflow. Visual elements are schematic and do not represent quantitative outputs. Image generated with GPT Image 2.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    ### Workflow
    1. **Network Setup**: upload a network, set the collapse target, and load the model.
    2. **Monitoring**: generate an attack sequence and run the simulation.
    3. **User Guide**: read the English or Chinese operating notes.
    """
)

st.markdown("### Page Shortcuts")
render_home_links()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### System Status")
    if st.session_state.initialized:
        st.markdown(
            '<div class="metric-card status-ready"><b>Status:</b> Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="metric-card status-pending"><b>Status:</b> Setup required</div>',
            unsafe_allow_html=True,
        )

with col2:
    st.markdown("### Network")
    if st.session_state.network:
        meta = st.session_state.network_metadata
        st.markdown(
            f"""
            <div class="metric-card">
            <b>Nodes:</b> {meta['num_nodes']}<br>
            <b>Edges:</b> {meta['num_edges']}<br>
            <b>Average degree:</b> {meta['avg_degree']:.2f}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No network loaded.")

with col3:
    st.markdown("### Model")
    if st.session_state.model:
        info = st.session_state.model.get_info()
        st.markdown(
            f"""
            <div class="metric-card">
            <b>Segments:</b> {info['num_segments']}<br>
            <b>Device:</b> {info['device']}<br>
            <b>Status:</b> Loaded
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No model loaded.")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: var(--app-muted);">
        <p>TCR-GIN is used as the predictive method; the application task is early warning for network structural collapse.</p>
        <p>
            <a href="https://github.com/DistinZhang/TCR-GIN" target="_blank" rel="noopener noreferrer" style="color: var(--app-accent); text-decoration: none;">
                GitHub: DistinZhang/TCR-GIN
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
