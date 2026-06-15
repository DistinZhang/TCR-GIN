#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Network setup page."""

import hashlib
import sys
import tempfile
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from webapp.core.network_loader import NetworkLoader  # noqa: E402
from webapp.core.validators import CollapseTargetValidator  # noqa: E402
from webapp.navigation import (  # noqa: E402
    init_session_state,
    render_appearance_selector,
    render_sidebar_nav,
)


st.set_page_config(page_title="Network Setup", page_icon="📊", layout="wide")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _model_dir_fingerprint(model_dir: str) -> str:
    """Fingerprint checkpoint names, sizes, and mtimes to invalidate stale cache."""
    root = Path(model_dir)
    digest = hashlib.sha256()
    if not root.exists():
        return "missing"
    for path in sorted(root.rglob("*.pt")):
        try:
            stat = path.stat()
            rel = str(path.relative_to(root)).replace("\\", "/")
            digest.update(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
        except OSError:
            continue
    return digest.hexdigest()


@st.cache_resource(show_spinner=False)
def _load_model_from_directory_cached(
    model_dir: str,
    config_hash: str,
    model_fingerprint: str,
    _config_bytes: bytes,
):
    from webapp.core.model_manager import ModelManager

    cache_key = hashlib.sha256(
        f"{Path(model_dir).resolve()}|{config_hash}|{model_fingerprint}".encode("utf-8")
    ).hexdigest()[:24]
    cache_dir = Path(tempfile.gettempdir()) / "tcrgin_webapp_model_cache" / f"local_{cache_key}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = cache_dir / "config.yaml"
    config_path.write_bytes(_config_bytes)

    model = ModelManager.load_from_directory(model_dir, str(config_path))
    is_valid, message = ModelManager.validate_config_match(model, str(config_path))
    if not is_valid:
        raise ValueError(message)
    return model, message


@st.cache_resource(show_spinner=False)
def _load_model_from_zip_cached(
    zip_hash: str,
    config_hash: str,
    _zip_bytes: bytes,
    _config_bytes: bytes,
):
    from webapp.core.model_manager import ModelManager

    cache_dir = Path(tempfile.gettempdir()) / "tcrgin_webapp_model_cache" / f"zip_{zip_hash[:16]}_{config_hash[:16]}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "model.zip"
    config_path = cache_dir / "config.yaml"
    if not zip_path.exists():
        zip_path.write_bytes(_zip_bytes)
    if not config_path.exists() or config_path.read_bytes() != _config_bytes:
        config_path.write_bytes(_config_bytes)

    model = ModelManager.load_from_zip(
        str(zip_path),
        str(config_path),
        temp_dir=str(cache_dir / "extracted"),
    )
    is_valid, message = ModelManager.validate_config_match(model, str(config_path))
    if not is_valid:
        raise ValueError(message)
    return model, message


init_session_state()
render_sidebar_nav()
render_appearance_selector()

if "last_network_upload_name" not in st.session_state:
    st.session_state.last_network_upload_name = None
if "last_model_signature" not in st.session_state:
    st.session_state.last_model_signature = None

DEMO_ROOT = project_root / "webapp" / "examples" / "transport_demo"
DEMO_NETWORK_PATH = DEMO_ROOT / "network" / "transport_multiplex_aggr_edges.npz"
DEMO_MODEL_DIR = DEMO_ROOT / "models"
DEMO_CONFIG_PATH = DEMO_ROOT / "config" / "transport_demo.yaml"

st.title("Network Setup")
st.caption("Load the network, define the collapse target, and attach a trained TCR-GIN model.")
st.markdown("---")

status_cols = st.columns(2)
with status_cols[0]:
    if st.session_state.get("network") is not None:
        meta = st.session_state.network_metadata
        st.success(
            f"Network loaded: {meta['num_nodes']} nodes and {meta['num_edges']} edges."
        )
    else:
        st.info("Network not loaded.")

with status_cols[1]:
    if st.session_state.get("model") is not None:
        info = st.session_state.model.get_info()
        st.success(
            f"Model loaded: {info['num_segments']} segment(s), ranges {', '.join(info['segments'])}."
        )
    else:
        st.info("Model not loaded.")

st.header("Bundled Demo")
st.caption("Load the packaged transport-network example, pretrained checkpoints, and YAML config.")
demo_available = DEMO_NETWORK_PATH.exists() and DEMO_MODEL_DIR.exists() and DEMO_CONFIG_PATH.exists()
load_demo_clicked = st.button(
    "Load Transport Demo",
    disabled=not demo_available,
    use_container_width=True,
)
if not demo_available:
    st.warning("Bundled demo files are missing. Check `webapp/examples/transport_demo/`.")

if load_demo_clicked and demo_available:
    try:
        from webapp.core.model_manager import ModelManager

        with st.spinner("Loading bundled transport demo..."):
            G, metadata = NetworkLoader.load_from_npz(str(DEMO_NETWORK_PATH))
            NetworkLoader.validate_network(G)
            model = ModelManager.load_from_directory(str(DEMO_MODEL_DIR), str(DEMO_CONFIG_PATH))
            is_valid, message = ModelManager.validate_config_match(model, str(DEMO_CONFIG_PATH))
            if not is_valid:
                raise ValueError(message)

        st.session_state.network = G
        st.session_state.network_metadata = metadata
        st.session_state.model = model
        st.session_state.collapse_target = 0.3
        st.session_state.simulator = None
        st.session_state.attack_sequence = None
        st.session_state.initialized = False
        st.session_state.warning_triggered = False
        st.session_state.collapsed = False
        st.session_state.last_network_upload_name = DEMO_NETWORK_PATH.name
        st.session_state.last_model_signature = ("Bundled Demo", str(DEMO_MODEL_DIR), str(DEMO_CONFIG_PATH))

        info = model.get_info()
        st.success(
            f"Transport demo loaded: {metadata['num_nodes']} nodes, "
            f"{metadata['num_edges']} edges, {info['num_segments']} model segment(s)."
        )
        st.info("Collapse target was set to 0.3. Open Monitoring to start an attack simulation.")
    except Exception as exc:
        st.error(f"Bundled demo loading failed: {exc}")

st.markdown("---")

st.header("Step 1: Import Network")
with st.expander("Network file format", expanded=False):
    st.markdown(
        """
        Upload a `.npz` file containing an edge list. The array may be stored
        under `edges`, `data`, or `edge_index`.

        Supported shapes:
        - `E x 2` edge list
        - `2 x E` PyG-style edge index
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

uploaded_network = st.file_uploader(
    "Upload network file (.npz)",
    type=["npz"],
    help="NumPy archive containing a network edge list.",
)

load_network_clicked = st.button(
    "Load Network",
    type="primary",
    disabled=uploaded_network is None,
    use_container_width=True,
)

if uploaded_network and not load_network_clicked and st.session_state.get("network") is None:
    st.caption("After choosing a file, click Load Network. Topology preview is disabled by default for faster setup.")

if uploaded_network and load_network_clicked:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp_file:
            tmp_file.write(uploaded_network.getvalue())
            tmp_path = tmp_file.name

        with st.spinner("Loading network..."):
            G, metadata = NetworkLoader.load_from_npz(tmp_path)
            NetworkLoader.validate_network(G)

        st.session_state.network = G
        st.session_state.network_metadata = metadata
        st.session_state.simulator = None
        st.session_state.attack_sequence = None
        st.session_state.initialized = False
        st.session_state.last_network_upload_name = uploaded_network.name

        st.success("Network loaded.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes", metadata["num_nodes"])
        col2.metric("Edges", metadata["num_edges"])
        col3.metric("Average Degree", f"{metadata['avg_degree']:.2f}")
        col4.metric("Connected", "Yes" if metadata["is_connected"] else "No")

        if not metadata["is_connected"]:
            st.warning(
                f"The network has {metadata['num_components']} connected components; "
                f"the largest component ratio is {metadata['largest_component_ratio']:.1%}."
            )
        if metadata.get("node_ids_relabelled"):
            st.info(
                "Node IDs were relabelled internally to contiguous integers 0..N-1. "
                "CSV attack sequences may still use the original IDs; they will be mapped automatically."
            )
    except Exception as exc:
        st.error(f"Network loading failed: {exc}")

if st.session_state.get("network") is not None:
    if st.checkbox("Show topology preview (slower)", value=False):
        from webapp.visualization.network_viz import create_network_visualization

        meta = st.session_state.network_metadata
        with st.spinner("Rendering topology..."):
            fig = create_network_visualization(
                st.session_state.network,
                title=f"Network topology - {meta['num_nodes']} nodes, {meta['num_edges']} edges",
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("Step 2: Set Collapse Target")

if st.session_state.network is None:
    st.warning("Import a network first.")
else:
    num_nodes = st.session_state.network_metadata["num_nodes"]
    st.markdown(
        f"""
        Current network size: **{num_nodes} nodes**.

        The collapse target defines when the largest connected component is
        considered collapsed. Enter either a ratio or an absolute LCC node count.
        """
    )

    target_value = st.number_input(
        "Collapse Target",
        min_value=0.0,
        max_value=float(num_nodes),
        value=0.3,
        step=0.01,
        help="Use a ratio in (0, 1), or an absolute node count in [1, N0].",
    )

    try:
        ratio, description = CollapseTargetValidator.parse_collapse_target(target_value, num_nodes)
        st.session_state.collapse_target = ratio
        st.success(f"Collapse target set to {description}.")
        st.progress(ratio, text=f"Collapse Target: {ratio:.1%}")
    except ValueError as exc:
        st.error(str(exc))

st.markdown("---")
st.header("Step 3: Load Model")

if st.session_state.network is None:
    st.warning("Import a network before loading a model.")
else:
    with st.expander("Model package format", expanded=False):
        st.markdown(
            """
            Provide a local model directory or a `.zip` archive, plus the YAML
            configuration used for training or evaluation. The loader maps model
            folders to node ranges from `model_suite[*].base_dir` or training
            `mix_id` entries. Folder names do not need to be pure numeric ranges.
            """
        )
        st.code(
            """transport-models/
├── 100-200-transport/
│   └── exp_001/
│       └── model_run_1.pt
└── 300-transport/
    └── exp_002/
        └── model_run_1.pt
""",
            language="text",
        )

    load_source = st.radio(
        "Model Source",
        ["Local Directory", "Upload ZIP"],
        horizontal=True,
        help="Use a local directory for local runs, or upload a ZIP archive for portable demos.",
    )

    col1, col2 = st.columns(2)
    with col1:
        if load_source == "Local Directory":
            model_dir = st.text_input(
                "Model Directory",
                value="",
                placeholder="Select or paste the root folder that contains model segments",
                help="Root directory containing one or more model segment folders.",
            )
            uploaded_model = None
        else:
            uploaded_model = st.file_uploader(
                "Upload model archive (.zip)",
                type=["zip"],
                help="ZIP archive containing segmented model checkpoints.",
            )
            model_dir = ""

    with col2:
        uploaded_config = st.file_uploader(
            "Upload configuration file (.yaml)",
            type=["yaml", "yml"],
            help="Training or evaluation YAML configuration for the model.",
        )

    can_load_model = uploaded_config and (
        (load_source == "Local Directory" and model_dir.strip())
        or (load_source == "Upload ZIP" and uploaded_model)
    )

    load_model_clicked = st.button(
        "Load Model",
        type="primary",
        disabled=not can_load_model,
        use_container_width=True,
    )

    if not load_model_clicked and st.session_state.get("model") is None:
        st.caption("Choose a model source and configuration, then click Load Model.")

    if load_model_clicked and can_load_model:
        try:
            config_bytes = uploaded_config.getvalue()
            config_hash = _sha256_bytes(config_bytes)

            with st.spinner("Loading model... Repeated loads of the same model use a cache."):
                if load_source == "Local Directory":
                    model, message = _load_model_from_directory_cached(
                        model_dir.strip(),
                        config_hash,
                        _model_dir_fingerprint(model_dir.strip()),
                        config_bytes,
                    )
                else:
                    zip_bytes = uploaded_model.getvalue()
                    model, message = _load_model_from_zip_cached(
                        _sha256_bytes(zip_bytes),
                        config_hash,
                        zip_bytes,
                        config_bytes,
                    )

            st.session_state.model = model
            st.session_state.simulator = None
            st.session_state.attack_sequence = None
            st.session_state.warning_triggered = False
            st.session_state.collapsed = False
            st.session_state.initialized = False
            st.session_state.model_load_count = st.session_state.get("model_load_count", 0) + 1
            st.session_state.last_model_signature = (
                load_source,
                "local-directory" if load_source == "Local Directory" else uploaded_model.name,
                uploaded_config.name,
            )

            info = model.get_info()
            st.success(message)
            st.caption(
                f"Model summary: {info['num_segments']} segment(s); device {info['device']}; "
                f"input dimension {info['config'].get('input_dim')}; ranges {', '.join(info['segments'])}."
            )

            with st.expander("Model Details", expanded=False):
                summary_cols = st.columns(4)
                summary_cols[0].metric("Load Count", st.session_state.model_load_count)
                summary_cols[1].metric("Segments", info["num_segments"])
                summary_cols[2].metric("Device", info["device"])
                summary_cols[3].metric("Input Dim", info["config"].get("input_dim"))

                segment_rows = [
                    {"Segment": segment, "Loaded": "✓"}
                    for segment in info["segments"]
                ]
                st.dataframe(segment_rows, use_container_width=True, hide_index=True)

                st.markdown("**Configuration**")
                config_cols = st.columns(4)
                config_cols[0].metric("Feature Dim", info["config"].get("feature_dim"))
                config_cols[1].metric("Hidden Dim", info["config"].get("hidden_dim"))
                config_cols[2].metric("Layers", info["config"].get("num_layers"))
                config_cols[3].metric("Dropout", info["config"].get("dropout"))

            if info["num_segments"] < 2:
                st.warning(
                    "Only one model segment was loaded. Check that the model source and YAML configuration "
                    "refer to every expected segment."
                )
            else:
                st.caption("Model segments were mapped from the configuration and loaded successfully.")
        except Exception as exc:
            st.error(f"Model loading failed: {exc}")

st.markdown("---")
st.header("Finalize")

if st.session_state.network and st.session_state.collapse_target and st.session_state.model:
    st.success("Setup is complete.")
    if st.button("Open Monitoring", type="primary"):
        st.session_state.initialized = True
        st.success("Initialization complete. Open the Monitoring page from the sidebar.")
else:
    missing = []
    if not st.session_state.network:
        missing.append("network")
    if not st.session_state.collapse_target:
        missing.append("collapse target")
    if not st.session_state.model:
        missing.append("model")
    st.warning(f"Still required: {', '.join(missing)}.")
