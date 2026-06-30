# Step-by-step: early warning of network breakdown with collapse distance

This Streamlit application provides early warning for network structural collapse under node-removal attacks, powered by TCR-GIN. It is designed for local analysis, reproducible demos, and GitHub release with the early-warning experiments.

Public deployment: [tcr-gin-early-warning.streamlit.app](https://tcr-gin-early-warning.streamlit.app/)

Walkthrough recording: [GitHub attachment](https://github.com/user-attachments/assets/74c7df14-a7b1-4e77-aa08-b13fcc22cfe0)

An archival copy is stored at [`assets/tcr-gin-early-warning.mp4`](assets/tcr-gin-early-warning.mp4).

## What It Does

- Load a `.npz` network from `edges`, `data`, or `edge_index`
- Load segmented TCR-GIN checkpoints from a local directory or ZIP archive
- Match model segments by the YAML configuration when `model_suite` is available
- Simulate random, degree-based, or uploaded attack sequences
- Monitor LCC Size, Collapse Distance, Natural Connectivity, and R(DCR)
- Trigger warning when `Collapse Distance <= k / N0`
- Export per-step metrics as CSV

## Run Locally

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

For a conda-based local environment:

```bash
conda env create -f webapp/environment.local.yml
conda activate tcr-gin-webapp
streamlit run webapp/app.py
```

## App Pages

- **Home**: application status and quick entry
- **Network Setup**: load network, set collapse target, and load model
- **Monitoring**: choose attacks, set Warning Target, run simulation, and inspect panels
- **User Guide**: English operating guide

The Network Setup page also includes a bundled transport demo. Click **Load
Transport Demo** to load the sample network, checkpoints, and YAML config
without uploading files.

## Network Format

Use a NumPy `.npz` file containing an edge list. Supported keys are `edges`, `data`, and `edge_index`. Both `E x 2` and PyG-style `2 x E` edge arrays are supported.

```python
import numpy as np

edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
])
np.savez("network_edges.npz", edges=edges)
```

## Model Format

The preferred workflow is to upload the YAML evaluation configuration used by the experiment scripts. The app reads `model_suite[*].node_range` and `model_suite[*].base_dir`, then finds matching checkpoints under the selected model directory or ZIP archive.

```yaml
model_suite:
  - name: "Base-Model-0-300"
    node_range: [0, 300]
    base_dir: "../models/transport-models/100-200-transport"

  - name: "Base-Model-300-400"
    node_range: [300, 400]
    base_dir: "../models/transport-models/300-transport"
```

Folder names do not need to be pure `start-end` ranges. When multiple `model_run_*.pt` files exist in one segment, the app selects `model_run_1.pt`, matching the early-warning scripts.

## Warning Target

The warning target is fixed to Collapse Distance. Enter an integer `k` in **Warning Target**. For the current single-node attack setting, `k` is equivalent to the number of attack steps. The app converts it to `k / N0` and triggers warning when:

```text
Collapse Distance <= k / N0
```

## Notes

- R(DCR) is recalculated on the current isolate-free residual graph using dynamic degree attack and current-remnant normalization.
- Collapse Distance follows the experiment logic: components larger than the Collapse Target are predicted separately, aggregated by node count, and normalized to the initial network scale.
- The displayed node-count distance is `Collapse Distance * N0`, rounded to the nearest integer.
- If no model prediction is available, Collapse Distance is recorded as `NaN` instead of using an LCC proxy.
