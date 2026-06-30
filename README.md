# TCR-GIN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20287619.svg)](https://doi.org/10.5281/zenodo.20287619)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](environment.yml)
[![Streamlit](https://img.shields.io/badge/Streamlit-demo-ff4b4b.svg)](webapp/)


**Physics-informed graph learning of collapse distance in complex networks**

Jie Zhang, Tao Wang, Yatai Ji, Hua He, Zhengqiu Zhu, Boquan Zhang, Xin Zhou, Changjun Fan,
Bin Chen, Manlio De Domenico, and Xin Lu.

Repository: [DistinZhang/TCR-GIN](https://github.com/DistinZhang/TCR-GIN)

![TCR-GIN conceptual workflow](webapp/assets/home_concept.png)

Conceptual illustration of the early-warning workflow. Visual elements are
schematic and do not represent quantitative outputs. Image generated with GPT
Image 2.

## Demo video

<video src="webapp/assets/tcr-gin-early-warning-readme.mp4" controls width="100%"></video>

This repository implements Topology-Consistency Regularized Graph Isomorphism
Network (TCR-GIN), a physics-informed graph learning framework for estimating
collapse distance in complex networks. Collapse distance is a configurable
safety-margin metric: it measures the minimum fraction of nodes that must be
removed to drive a network to a prescribed collapse target. TCR-GIN learns this
quantity from dismantling-derived upper-bound supervision and degradation-law
regularization, enabling fast graph-level prediction, trajectory monitoring
under sustained attacks, and early-warning signals for network breakdown.

The code accompanies the Nature Communications formatted manuscript and its
Supplementary Information. The manuscript reports synthetic-network benchmarks,
real-world network tests, trajectory-property analyses, early-warning
experiments, collapse-profile prediction, and comparisons against 28 baseline
methods.

## Contents

- [Quick links](#quick-links)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data, models, and results](#data-models-and-results)
- [Reproducing experiments](#reproducing-experiments)
- [Manuscript figures](#manuscript-figures)
- [Interactive web application](#interactive-web-application)
- [Graph and model formats](#graph-and-model-formats)
- [Reproducibility notes](#reproducibility-notes)
- [License](#license)

## Quick links

- **Code**: [GitHub repository](https://github.com/DistinZhang/TCR-GIN)
- **Complete experimental package**: [Zenodo archive](https://doi.org/10.5281/zenodo.20287619)
- **Interactive demo**: [tcr-gin-early-warning.streamlit.app](https://tcr-gin-early-warning.streamlit.app/)
- **Demo video**: [GitHub attachment](https://github.com/user-attachments/assets/74c7df14-a7b1-4e77-aa08-b13fcc22cfe0)
- **Citation**: use `CITATION.cff` and cite the associated manuscript and
  Zenodo record.

## Repository layout

```text
TCR-GIN/
├── Figures/                      # Main manuscript figures in PDF format
├── assets/                       # Lightweight README/project-page assets
├── arguments.py                  # Core scalar TCR-GIN argument parser
├── data_loader.py                # Graph triplet loading and dataloader logic
├── train_piss.py                 # Core training routine
├── run_experiments.py            # YAML-driven experiment launcher
├── run.sh                        # Master reproduction command list
├── requirements.txt              # Core pip environment
├── environment.yml               # Core conda environment
├── .streamlit/                   # Streamlit Cloud theme/runtime settings
├── CITATION.cff                  # Citation metadata
├── model/
│   └── tcr_gin.py                # TCR-GIN model definition
├── configs/                      # Core training YAML files
├── experiments/
│   ├── introduction/             # Research Status Analysis
│   ├── baseline_comparison/      # Accuracy, runtime, exact-label, ablation, sensitivity tests
│   ├── trajectory_analysis/      # Monotonicity, smoothness, additivity trajectory analyses
│   ├── early_warning/            # Early-warning and robustness-metric comparisons
│   └── collapse_profile/         # Discrete collapse-profile prediction experiments
├── utils/                        # Data preparation, feature generation, split and aggregation tools
└── webapp/                       # Streamlit early-warning application
```

Large datasets, trained checkpoints, generated results, logs, and cache files
are not intended to be stored in GitHub. They are distributed through the
external archive described below.

## Installation

The current server environment used during release preparation is:

```text
Python 3.10
CUDA 11.8
torch 2.1.2+cu118
torch-geometric 2.6.1
```

### Option 1: conda

```bash
conda env create -f environment.yml
conda activate tcr-gin
```

### Option 2: pip

Create and activate a Python 3.10 environment, then run:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes CUDA 11.8 wheel locations for the PyTorch and
PyTorch Geometric stack used on the current server. If you use a different CUDA
or CPU-only environment, install `torch`, `torch-geometric`, and the required
PyG extension wheels following the official PyTorch/PyG instructions for your
platform, then install the remaining packages.

### Web application environment

The Streamlit web application has a small number of additional dependencies.
Install them from the repository root with:

```bash
pip install -r webapp/requirements.txt
```

or create the dedicated local conda environment:

```bash
conda env create -f webapp/environment.local.yml
conda activate tcr-gin-webapp
```

`webapp/requirements.txt` is CPU-friendly for Streamlit Community Cloud.
`webapp/environment.local.yml` is the CUDA-enabled local/server environment.

## Data, models, and results

The datasets, trained model checkpoints, generated result tables, and large
intermediate artifacts used to reproduce the manuscript are available from
[Zenodo](https://doi.org/10.5281/zenodo.20287619).

The manuscript Data Availability statement points to the same record.

The GitHub repository is the script-focused public code release. The Zenodo
archive preserves a complete release snapshot: the external datasets are stored
under `data/`, while the code, retained model checkpoints, generated experiment
outputs, Streamlit app assets, and documentation are stored together under
`TCR-GIN/`. After downloading the archive, unpack it as a single archive root or
update the relevant YAML files and `run.sh` paths to match your local layout.

```text
zenodo/
├── data/
│   ├── split/           # Graph splits
│   ├── split-exact/     # Exact-label data
│   ├── split_111/       # Transport tau=0.3
│   ├── split_185/       # Transport tau=0.5
│   ├── split_353/       # Power tau=0.3
│   ├── split_588/       # Power tau=0.5
│   ├── data_synth/      # Synthetic data
│   ├── data_real/       # Real data
│   ├── data_trajectory/ # Trajectory data
│   ├── data_metric/     # Metric data
│   └── profile/         # Profile data
├── TCR-GIN/
│   ├── configs/         # Config files
│   ├── experiments/     # Scripts and outputs
│   ├── model/           # Model code
│   ├── models/          # Checkpoints
│   ├── utils/           # Utility scripts
│   ├── webapp/          # Streamlit app
│   └── README.md        # Usage guide
├── MANIFEST.md          # File inventory
├── THIRD_PARTY_SOURCES.md
├── CITATION.cff
└── checksums_sha256.txt
```

If your local directory structure differs, update the relevant paths in:

- `configs/*.yaml`
- `experiments/*/configs/*.yaml`
- `run.sh`

For integrity checking, use the checksum file included in the external archive:

```bash
sha256sum -c checksums_sha256.txt
```

## Reproducing experiments

The complete command list is maintained in `run.sh`. It is the authoritative
entry point for reproducing the manuscript workflows.

```bash
chmod +x run.sh
./run.sh
```

Most commands in `run.sh` are intentionally commented out because many steps are
expensive or only need to be run once. Edit the script and uncomment the blocks
you want to run after the Zenodo archive has been downloaded and all paths have
been configured. In the current working copy, the collapse-profile evaluation
commands at the end of `run.sh` are active; comment them out if you want to use
the file only as a command reference.

The script is organized as follows:

```text
STEP 0  Data preparation
        Consolidate heuristic results, generate observation labels, and
        aggregate remnant/component datasets.

STEP 1  Model training
        Train baseline-comparison, trajectory-analysis, ablation, sensitivity,
        and early-warning model suites from YAML configs.

STEP 2  Introduction experiment
        Generate example networks, run targeted attacks, compute metrics, and
        call experiments/introduction/plot1.py.

STEP 3  Baseline comparison
        Evaluate checkpoints, compare against baselines, run exact and
        observation-label comparisons, and generate baseline-comparison figures.

STEP 4  Trajectory analysis
        Run trajectory-property tests, ablation tests, sensitivity tests, and
        generate trajectory-analysis figures.

STEP 5  Early-warning signal and metric comparison
        Run early-warning property tests, decision-window analysis, classical
        robustness-metric comparison, and generate early-warning figures.

STEP 6  Collapse-profile experiments
        Train/evaluate collapse-profile prediction and generate profile plots.
```

Representative commands include:

```bash
# Core model training
python run_experiments.py --config configs/train-multisource.yaml
python run_experiments.py --config configs/train-multisource-exact.yaml
python run_experiments.py --config configs/train-ablation.yaml
python run_experiments.py --config configs/train-sensitivity.yaml

# Baseline comparison
python experiments/baseline_comparison/test_performance.py \
  --config experiments/baseline_comparison/configs/test_multisource-LBWE-CPU.yaml
python experiments/baseline_comparison/test_performance.py \
  --config experiments/baseline_comparison/configs/test_multisource-LBWE-GPU.yaml
python experiments/baseline_comparison/plot3.py

# Trajectory analysis
python experiments/trajectory_analysis/test_properties.py \
  --config experiments/trajectory_analysis/configs/test_properties_base_multisource-transport.yaml
python experiments/trajectory_analysis/test_properties.py \
  --config experiments/trajectory_analysis/configs/test_properties_base_multisource-power.yaml
python experiments/trajectory_analysis/plot4.py

# Early-warning analysis
python experiments/early_warning/test_properties.py \
  --config experiments/early_warning/configs/transport-111.yaml
python experiments/early_warning/test_properties.py \
  --config experiments/early_warning/configs/transport-185.yaml
python experiments/early_warning/test_properties.py \
  --config experiments/early_warning/configs/power-353.yaml
python experiments/early_warning/test_properties.py \
  --config experiments/early_warning/configs/power-588.yaml
python experiments/early_warning/plot7.py

# Collapse-profile analysis
python experiments/collapse_profile/test_profile.py \
  --config experiments/collapse_profile/configs/test/test_reddit_profile.yaml
python experiments/collapse_profile/plot_profile.py \
  --config experiments/collapse_profile/configs/plot/plot_reddit_profile.yaml
```

## Manuscript figures

The main manuscript figures are included in `Figures/` as PDF files:

```text
Figures/
├── figure_1.pdf
├── figure_2.pdf
├── figure_3.pdf
├── figure_4.pdf
├── figure_5.pdf
├── figure_6.pdf
└── figure_7.pdf
```

The README hero image is `webapp/assets/home_concept.png`, a schematic
conceptual illustration for the public demo landing page. The manuscript source
figure files remain in `Figures/`.

## Interactive web application

The repository includes a Streamlit application for interactive early-warning
analysis under node-removal attacks. It can load a network, load segmented
TCR-GIN checkpoints with their YAML configuration, simulate attacks, and track:

- LCC size
- Collapse distance
- Natural connectivity
- R(DCR)

The app includes a bundled transport demo under
`webapp/examples/transport_demo/`. On the **Network Setup** page, click **Load
Transport Demo** to load the sample network, model checkpoints, and YAML config
without downloading external data.

Public Streamlit deployment: [tcr-gin-early-warning.streamlit.app](https://tcr-gin-early-warning.streamlit.app/)

Recorded walkthrough: [GitHub attachment](https://github.com/user-attachments/assets/74c7df14-a7b1-4e77-aa08-b13fcc22cfe0)

An archival copy is stored at [`webapp/assets/tcr-gin-early-warning.mp4`](webapp/assets/tcr-gin-early-warning.mp4).

### Start locally

From the repository root:

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

The local URL is usually:

```text
http://localhost:8501
```

### Start on a server

```bash
streamlit run webapp/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

Open the server URL shown by Streamlit, for example:

```text
http://<server-ip>:8501
```

### Webapp workflow

1. Open **Network Setup**.
2. For a quick demo, click **Load Transport Demo**. For custom data, upload a
   `.npz` network file.
3. Set the collapse target. You may enter a ratio such as `0.3` or an absolute
   LCC size such as `150`.
4. Load a model directory or ZIP archive containing `model_run_*.pt`
   checkpoints.
5. Upload the YAML configuration used by the matching experiment. The app reads
   `model_suite[*].node_range` and `model_suite[*].base_dir` to map checkpoints
   to network-size ranges.
6. Open **Monitoring**.
7. Choose an attack sequence: random, degree-based, or uploaded CSV.
8. Set **Warning Target**. For single-node attacks, an integer `k` is converted
   to `k / N0`, where `N0` is the initial number of nodes.
9. Run the simulation manually or automatically.
10. Export the per-step metrics as CSV if needed.

The warning condition used by the app is:

```text
Collapse Distance <= k / N0
```

Collapse-distance predictions in the webapp are clamped to `[0, 1]` after label
rescaling, matching the evaluation scripts.

## Graph and model formats

Most graph samples use a three-file convention:

```text
<graph_id>_edges.npz
<graph_id>_features.npy
<graph_id>_label.json
```

The edge archive generally contains an `edges` array with shape `(num_edges, 2)`.
The feature file stores a node-feature matrix. The label JSON stores graph
metadata and target values such as `critical_threshold`.

The webapp accepts a single `.npz` file containing one of the following keys:

```text
edges
data
edge_index
```

Both edge-list shape `E x 2` and PyTorch Geometric style `2 x E` are supported.

Model checkpoint folders typically follow this pattern:

```text
<segment-name>/
└── exp_001/
    ├── model_run_1.pt
    ├── model_run_2.pt
    └── ...
```

When multiple runs are present in one segment, the webapp selects
`model_run_1.pt` by default.

## Reproducibility notes

- Set random seeds in YAML files where supported.
- GPU determinism may vary across CUDA, cuDNN, PyTorch, and driver versions.
- Some exact-label and dismantling computations are expensive; GitHub keeps a
  script-focused code release, while Zenodo preserves the complete data,
  checkpoint, result, and app snapshot.
- Path assumptions in older experiment commands may reflect the original server
  layout. Update YAML files and `run.sh` paths if your data archive is unpacked
  elsewhere.
- Generated logs, caches, large checkpoints, model folders, raw outputs, and
  notebook checkpoint directories should not be committed to GitHub. The small
  `webapp/examples/transport_demo/` assets are intentionally kept for the
  public Streamlit demo.

## License

The source code in this repository is released under the MIT License. See
`LICENSE`.

The Zenodo archive contains third-party-derived network data and derived
artifacts. Those data are subject to their upstream licenses and attribution
requirements. Consult the archive-level `LICENSE` and `THIRD_PARTY_SOURCES.md`
before redistributing or reusing the data.
