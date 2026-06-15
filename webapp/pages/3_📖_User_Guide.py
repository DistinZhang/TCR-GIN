#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
User guide page for the TCR-GIN web application.
"""

import streamlit as st

from webapp.navigation import init_session_state, render_appearance_selector, render_sidebar_nav


st.set_page_config(page_title="User Guide", page_icon="📖", layout="wide")
init_session_state()
render_sidebar_nav()
render_appearance_selector()

st.title("User Guide / 用户手册")

language = st.radio("Language / 语言", ["English", "中文"], horizontal=True)


def english_guide():
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
        graph has multiple connected components, components larger than the
        collapse target are predicted separately and aggregated by node count.
        The final Collapse Distance is normalized to the initial network scale,
        so the node-count distance shown in the status area is
        `Collapse Distance * N0`.
        """
    )

    st.header("Network Setup")
    st.markdown(
        """
        1. Upload a `.npz` network file.
        2. The file may contain `edges`, `data`, or `edge_index`.
        3. Supported edge shapes are `E x 2` and PyG-style `2 x E`.
        4. Set the collapse target as either a ratio, such as `0.3`, or an
           absolute LCC node count, such as `150`.
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
        The model loader uses the YAML configuration to map model folders to
        node ranges. Folder names do not need to be pure ranges. For example,
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
        If multiple `model_run_*.pt` files are present in the same segment, the
        app selects `model_run_1.pt`, matching the experiment scripts.
        """
    )

    st.header("Real-Time Monitoring")
    st.markdown(
        """
        1. Choose an attack sequence: random, degree-based, or CSV upload.
        2. Set **Warning Target** as an integer. In single-node attacks, this is
           equivalent to the number of attack steps. The app displays the
           converted ratio `target / N0`.
        3. Initialize the simulator.
        4. Run one attack step manually or use automatic mode.

        The Collapse Distance panel marks the decision threshold and highlights the
        warning point once the warning condition is reached.
        """
    )

    st.header("Output")
    st.markdown(
        """
        After running an attack, download the CSV file to inspect per-step
        metrics: removed node, LCC, Collapse Distance, natural connectivity, and
        R(DCR).
        """
    )


def chinese_guide():
    st.header("系统概述")
    st.markdown(
        """
        TCR-GIN 用于模拟网络在节点移除攻击下的结构崩溃早期预警状态，并显示四类预警指标：
        LCC Size、Collapse Distance、自然连通度和 R(DCR)。

        实时预警规则与实验脚本保持一致：用户输入整数预警目标 `k`，系统自动转换为
        `k / N0`，其中 `N0` 是初始网络节点数。当 `Collapse Distance <= k / N0`
        时触发预警。

        Collapse Distance 与实验脚本逻辑一致：如果当前残余网络已经分裂成多个连通片，
        系统会对规模仍大于 Collapse Target 的连通片分别预测，再按连通片节点数
        加权整合并校正到初始网络尺度。因此状态区显示的“距离崩溃”节点数为
        `Collapse Distance * N0` 取整。
        """
    )

    st.header("网络初始化")
    st.markdown(
        """
        1. 上传 `.npz` 网络文件。
        2. 文件可包含 `edges`、`data` 或 `edge_index`。
        3. 支持 `E x 2` 边表，也支持 PyG 风格的 `2 x E`。
        4. 设置崩溃目标：可输入比例，例如 `0.3`；也可输入 LCC 绝对节点数，例如 `150`。
        5. 从本地目录或 ZIP 加载 TCR-GIN 模型，并上传匹配的 YAML 配置。
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

    st.header("模型加载")
    st.markdown(
        """
        模型加载器会根据 YAML 配置把模型文件夹映射到节点范围。文件夹名称不必是纯数字范围。
        例如 `100-200-transport` 可以在配置中映射到 `[0, 300]`：
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
        如果同一个模型段中存在多个 `model_run_*.pt`，网页默认选择 `model_run_1.pt`，
        与实验脚本保持一致。
        """
    )

    st.header("实时监控")
    st.markdown(
        """
        1. 选择攻击序列：随机攻击、基于度的攻击或上传 CSV。
        2. 设置 **Warning Target**。在单节点攻击场景中，该整数等价于攻击步数；
           系统会自动显示转换后的比例 `target / N0`。
        3. 初始化模拟器。
        4. 手动执行单步攻击，或使用自动模式连续攻击。

        崩溃距离面板会显示决策阈值，并在达到预警条件后标记预警点。
        """
    )

    st.header("结果导出")
    st.markdown(
        """
        攻击开始后，可以下载 CSV 文件，查看每一步的移除节点、LCC、Collapse Distance、
        自然连通度和 R(DCR)。
        """
    )


if language == "English":
    english_guide()
else:
    chinese_guide()
