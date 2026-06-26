# Web Application Summary

This directory contains the Streamlit application for early warning of network structural collapse under node-removal attacks.

## Implemented Components

- `app.py`: Home page, status overview, and page shortcuts.
- `pages/1_Network_Setup.py`: network import, collapse target setup, model loading, and setup summary.
- `pages/2_Real_Time.py`: attack sequence control, Warning Target input, simulation controls, monitoring panels, network view, and CSV export.
- `pages/3_User_Guide.py`: English user guide matching the current UI.
- `core/network_loader.py`: `.npz` network loading from `edges`, `data`, or `edge_index`.
- `core/model_manager.py`: segmented TCR-GIN model loading from local directories or ZIP archives.
- `core/attack_simulator.py`: attack simulation and metric calculation.
- `visualization/panels.py`: LCC Size, Collapse Distance, Natural Connectivity, and R(DCR) panels.
- `visualization/network_viz.py`: interactive network visualization with cached fixed layouts.

## Scientific Logic

- Collapse Distance follows the experiment logic: components larger than the Collapse Target are predicted separately, aggregated by node count, and normalized to the initial network scale.
- The displayed node-count distance is `Collapse Distance * N0`, rounded to the nearest integer.
- R(DCR) uses dynamic degree attack on the current isolate-free residual graph and normalizes by the current residual graph size.
- If model inference is unavailable, Collapse Distance is stored as `NaN`; the app does not use LCC as a substitute prediction.
- Warning Target is an integer `k` converted to `k / N0`; warning triggers when `Collapse Distance <= k / N0`.

## Model Loading

When `model_suite` is available, the app uses `node_range` and `base_dir` from the YAML config. Example generic mapping:

- `100-200-transport` -> `[0, 300]`
- `300-transport` -> `[300, 400]`

If `model_suite` is absent, model-family directories are inferred from checkpoint paths and normalized into non-overlapping ranges.
