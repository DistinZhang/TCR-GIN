# Early Warning System for Network Structural Collapse

This branch hosts the standalone Streamlit web application for interactive
early-warning analysis of network structural collapse with TCR-GIN.

The full research code, reproduction scripts, complete datasets, trained model
collections, and result artifacts are maintained on the `main` branch and in
the associated Zenodo archive.

## Online App

Open the deployed Streamlit app:

[https://early-warning-collapse.streamlit.app](https://early-warning-collapse.streamlit.app)

## What the App Does

- Loads a network edge-list file in `.npz` format.
- Loads trained TCR-GIN checkpoints and the matching YAML configuration.
- Simulates node-removal attacks on the loaded network.
- Predicts collapse distance during the attack process.
- Visualizes early-warning indicators including LCC size, collapse distance,
  natural connectivity, and R(DCR).
- Provides a bundled transport-network demo for immediate testing.

## Quick Start in the Online App

1. Open the Streamlit app.
2. Go to **Network Setup**.
3. Click **Load Transport Demo**.
4. Confirm that both the network and the model are loaded.
5. Open **Monitoring**.
6. Generate or upload an attack sequence.
7. Start the simulation and inspect the warning panels.

## Bundled Transport Demo

The deployment branch includes a compact demo package:

```text
webapp/examples/transport_demo/
├── network/
│   └── transport_multiplex_aggr_edges.npz
├── config/
│   └── transport_demo.yaml
└── models/
    ├── 100-200-transport+LBWE/exp_001/model_run_1.pt
    └── 300-transport+LBWE/exp_002/model_run_1.pt
```

These files are included only to make the online app immediately usable. They
are not the complete experiment data release.

## Local Run

From this branch, install the lightweight webapp environment and start
Streamlit:

```bash
pip install -r requirements.txt
streamlit run webapp/app.py
```

## Streamlit Community Cloud Settings

Use the following deployment settings:

```text
Repository: DistinZhang/TCR-GIN
Branch: streamlit-demo
Main file path: webapp/app.py
```

The app is standalone on this branch. The TCR-GIN model definition required for
inference is included under:

```text
webapp/core/tcr_gin.py
```

## Relation to the Main Repository

Use the `main` branch for the manuscript-level repository, including the full
README, environment files, experiment commands, reproduction scripts, and
Zenodo DOI. Use this `streamlit-demo` branch only for the online web
application deployment.
