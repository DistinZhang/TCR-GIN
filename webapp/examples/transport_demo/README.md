# Transport Demo Assets

This directory contains the bundled example used by the Streamlit web
application. It lets readers open the app, click **Load Transport Demo**, and
run a small early-warning simulation without downloading the full Zenodo
archive.

## Included Files

- `network/transport_multiplex_aggr_edges.npz`
  - Edge-list file for the transport multiplex aggregate network.
  - The NumPy archive contains one array named `edges`.
  - Shape: `(430, 2)`.
  - Node IDs: integer IDs from `0` to `368`.

- `config/transport_demo.yaml`
  - Training-style YAML config used by the webapp model loader.
  - The config defines two demo model families:
    `100-200-transport+LBWE` and `300-transport+LBWE`.
  - Model parameters use `feature_dim: 3`, `hidden_dim: 128`,
    `num_layers: 5`, `jk_type: lstm`, residual connections, virtual node, and
    GELU activation.

- `models/100-200-transport+LBWE/exp_001/model_run_1.pt`
  - Checkpoint for the lower-size model segment inferred from the config.

- `models/300-transport+LBWE/exp_002/model_run_1.pt`
  - Checkpoint for the upper-size model segment inferred from the config.

## Purpose

These files are only a compact webapp demonstration package. They are not the
complete experiment data release. Full datasets, trained model collections, all
model runs, and generated result artifacts are distributed through the Zenodo
archive referenced in the repository-level README.

## Licensing Note

The transport-network sample is provided only for webapp demonstration and
deployment testing. Follow the third-party source and license notes in the
Zenodo archive before redistributing or reusing the data outside this demo
context.
