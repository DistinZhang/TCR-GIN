#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared Streamlit navigation helpers."""

import streamlit as st


PAGES = [
    ("Home", "app.py"),
    ("Network Setup", "pages/1_Network_Setup.py"),
    ("Monitoring", "pages/2_Real_Time.py"),
    ("User Guide", "pages/3_User_Guide.py"),
]

def init_session_state() -> None:
    """Initialize shared session keys used across pages."""
    defaults = {
        "initialized": False,
        "network": None,
        "network_metadata": None,
        "model": None,
        "collapse_target": None,
        "simulator": None,
        "attack_sequence": None,
        "is_attacking": False,
        "is_auto_attacking": False,
        "warning_triggered": False,
        "collapsed": False,
        "appearance_mode": "Dark",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_appearance_selector() -> None:
    """Render the shared light/dark/system appearance control."""
    current_mode = st.session_state.get("appearance_mode", "Dark")
    if current_mode not in ["Light", "Dark", "System"]:
        current_mode = "Dark"
    st.session_state.appearance_mode = current_mode
    with st.sidebar:
        mode = st.radio(
            "Appearance",
            ["Light", "Dark", "System"],
            index=["Light", "Dark", "System"].index(current_mode),
            horizontal=False,
            help="Switch the interface and chart appearance. System follows the browser or OS preference."
        )
        st.session_state.appearance_mode = mode
    apply_appearance_css(mode)


def apply_appearance_css(mode: str) -> None:
    """Apply CSS variables for the selected appearance mode."""
    dark_css = """
        --app-bg: #0f141b;
        --app-surface: #151b23;
        --app-surface-2: #1f2935;
        --app-text: #f3f6fa;
        --app-muted: #b5c0cf;
        --app-border: #303b49;
        --app-accent: #6ea8fe;
        --app-accent-soft: rgba(110, 168, 254, 0.16);
        --app-shadow: rgba(0, 0, 0, 0.25);
    """
    light_css = """
        --app-bg: #f7f8fb;
        --app-surface: #ffffff;
        --app-surface-2: #eef2f7;
        --app-text: #172033;
        --app-muted: #5b687c;
        --app-border: #d8dee9;
        --app-accent: #2764c5;
        --app-accent-soft: rgba(39, 100, 197, 0.10);
        --app-shadow: rgba(16, 24, 40, 0.04);
    """

    if mode == "System":
        root_vars = light_css
        media_css = f"""
            @media (prefers-color-scheme: dark) {{
                :root {{
                    {dark_css}
                }}
            }}
        """
    elif mode == "Dark":
        root_vars = dark_css
        media_css = ""
    else:
        root_vars = light_css
        media_css = ""

    st.markdown(
        f"""
        <style>
            :root {{
                {root_vars}
            }}
            {media_css}
            .stApp {{
                background: var(--app-bg);
                color: var(--app-text);
            }}
            [data-testid="stSidebar"], [data-testid="stHeader"] {{
                background: var(--app-surface);
            }}
            [data-testid="stToolbar"], [data-testid="stDecoration"] {{
                color: var(--app-text);
            }}
            [data-testid="stAppViewContainer"] {{
                background:
                    radial-gradient(circle at 18% 0%, color-mix(in srgb, var(--app-accent) 18%, transparent) 0, transparent 28rem),
                    var(--app-bg);
            }}
            [data-testid="stSidebar"] * {{
                color: var(--app-text);
            }}
            .hero-band, .metric-card, div[data-testid="metric-container"] {{
                background: var(--app-surface) !important;
                border-color: var(--app-border) !important;
                color: var(--app-text) !important;
                box-shadow: 0 1px 2px var(--app-shadow);
            }}
            h1, h2, h3, h4, h5, h6, p, span, label, li,
            [data-testid="stMarkdownContainer"],
            [data-testid="stWidgetLabel"],
            [data-testid="stMetricLabel"],
            [data-testid="stMetricValue"],
            [data-testid="stMetricDelta"],
            [data-testid="stCaptionContainer"] {{
                color: var(--app-text) !important;
            }}
            small, .stCaption {{
                color: var(--app-muted) !important;
            }}
            a {{
                color: var(--app-accent) !important;
            }}
            [data-baseweb="input"] input,
            [data-baseweb="select"] > div,
            [data-baseweb="textarea"] textarea,
            [data-baseweb="popover"],
            [data-baseweb="menu"],
            [data-baseweb="menu"] ul,
            [role="listbox"],
            [role="option"],
            div[data-testid="stNumberInput"] input,
            div[data-testid="stTextInput"] input,
            div[data-testid="stFileUploader"] section {{
                background: var(--app-surface) !important;
                color: var(--app-text) !important;
                border-color: var(--app-border) !important;
            }}
            [data-baseweb="popover"] *,
            [data-baseweb="menu"] *,
            [role="listbox"] *,
            [role="option"] * {{
                color: var(--app-text) !important;
            }}
            [role="option"]:hover {{
                background: var(--app-accent-soft) !important;
            }}
            [data-baseweb="select"] span,
            [data-baseweb="input"] input::placeholder {{
                color: var(--app-muted) !important;
            }}
            div[data-testid="stFileUploader"] small,
            div[data-testid="stFileUploader"] span {{
                color: var(--app-muted) !important;
            }}
            div[data-testid="stRadio"] label,
            div[data-testid="stCheckbox"] label {{
                color: var(--app-text) !important;
            }}
            div[data-testid="stRadio"] div[role="radiogroup"] label p,
            div[data-testid="stCheckbox"] label p {{
                color: var(--app-text) !important;
            }}
            [data-testid="stHorizontalBlock"] {{
                gap: 1rem;
            }}
            div[data-testid="stExpander"],
            details,
            div[data-testid="stJson"],
            div[data-testid="stDataFrame"],
            div[data-testid="stTable"],
            div[data-testid="stDownloadButton"] {{
                background: var(--app-surface) !important;
                color: var(--app-text) !important;
                border-color: var(--app-border) !important;
            }}
            div[data-testid="stAlert"] {{
                color: var(--app-text) !important;
            }}
            div[data-testid="stAlert"] * {{
                color: inherit !important;
            }}
            div[data-testid="stDataFrame"] * {{
                color: var(--app-text) !important;
            }}
            div[data-testid="stDataFrame"] [role="gridcell"],
            div[data-testid="stDataFrame"] [role="columnheader"] {{
                background: var(--app-surface-2) !important;
            }}
            div[data-testid="stTable"] table,
            div[data-testid="stTable"] th,
            div[data-testid="stTable"] td {{
                background: var(--app-surface) !important;
                color: var(--app-text) !important;
                border-color: var(--app-border) !important;
            }}
            div[data-testid="stTabs"] button,
            div[data-testid="stTabs"] button p {{
                color: var(--app-text) !important;
            }}
            div[data-testid="stTabs"] [aria-selected="true"] {{
                border-bottom-color: var(--app-accent) !important;
            }}
            div[data-testid="stJson"] pre,
            div[data-testid="stJson"] span,
            div[data-testid="stJson"] code {{
                background: var(--app-surface-2) !important;
                color: var(--app-text) !important;
                opacity: 1 !important;
            }}
            div[data-testid="stJson"] svg {{
                fill: var(--app-text) !important;
                color: var(--app-text) !important;
            }}
            .st-emotion-cache-1v0mbdj,
            .st-emotion-cache-1avcm0n {{
                color: var(--app-text) !important;
            }}
            button {{
                border-color: var(--app-border) !important;
            }}
            button[kind="secondary"], .stButton > button {{
                background: var(--app-surface) !important;
                color: var(--app-text) !important;
            }}
            button[kind="primary"], [data-testid="stBaseButton-primary"] {{
                background: linear-gradient(135deg, var(--app-accent), color-mix(in srgb, var(--app-accent) 65%, #2dd4bf 35%)) !important;
                color: #ffffff !important;
                border: 0 !important;
            }}
            button[kind="primary"] *, [data-testid="stBaseButton-primary"] * {{
                color: #ffffff !important;
            }}
            .stButton > button,
            [data-testid="stBaseButton-secondary"],
            [data-testid="stBaseButton-primary"] {{
                border-radius: 8px !important;
            }}
            .stButton > button:hover {{
                background: var(--app-accent-soft) !important;
                border-color: var(--app-accent) !important;
                color: var(--app-text) !important;
            }}
            code, pre {{
                background: var(--app-surface-2) !important;
                color: var(--app-text) !important;
            }}
            .success-box, .warning-box, .error-box {{
                color: var(--app-text) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav() -> None:
    """Render stable navigation labels."""
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] .stPageLink {
                border-radius: 8px;
            }
            section[data-testid="stSidebar"] [data-testid="stPageLink"] a,
            section[data-testid="stSidebar"] a {
                color: var(--app-text) !important;
                text-decoration: none;
            }
            section[data-testid="stSidebar"] [data-testid="stPageLink"] {
                border-radius: 8px;
            }
            section[data-testid="stSidebar"] [data-testid="stPageLink"][aria-current="page"] {
                background: var(--app-surface-2);
            }
            section[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {
                background: var(--app-accent-soft);
            }
            .block-container {
                padding-top: 1.75rem;
                padding-bottom: 2rem;
            }
            h1, h2, h3 {
                letter-spacing: 0;
            }
            div[data-testid="metric-container"] {
                background: var(--app-surface);
                border: 1px solid var(--app-border);
                border-radius: 8px;
                padding: 0.9rem 1rem;
                box-shadow: 0 1px 2px var(--app-shadow);
            }
            div[data-testid="stAlert"] {
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Navigation")
        if hasattr(st, "page_link"):
            try:
                for label, page_path in PAGES:
                    st.page_link(page_path, label=label)
            except Exception:
                for label, page_path in PAGES:
                    if st.button(label, key=f"nav_{page_path}", use_container_width=True):
                        st.switch_page(page_path)
        else:
            st.warning("The installed Streamlit version is too old for page links. Please use streamlit>=1.31.")
        st.divider()


def render_home_links() -> None:
    """Render prominent page links in the main page body."""
    if hasattr(st, "page_link"):
        cols = st.columns(3)
        for col, (label, page_path) in zip(cols, PAGES[1:]):
            with col:
                st.page_link(page_path, label=label)
