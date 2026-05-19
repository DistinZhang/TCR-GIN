# TCR-GIN

TCR-GIN is a graph neural network framework for predicting collapse-related
properties of complex networks during dismantling processes. The repository
contains training, evaluation, robustness analysis, trajectory-property analysis,
collapse-profile prediction, and figure-generation scripts used for reproducible
experiments.

## Repository Layout

- `arguments.py`, `data_loader.py`, `train_piss.py`, `run_experiments.py`: core
  training pipeline for scalar TCR-GIN experiments.
- `model/tcr_gin.py`: TCR-GIN model definition.
- `configs/`: YAML files for training campaigns.
- `experiments/baseline_comparison/`: baseline comparison and performance plots.
- `experiments/trajectory_analysis/`: trajectory-property tests and plots.
- `experiments/early_warning/`: early-warning robustness experiments.
- `experiments/collapse_profile/`: collapse-profile prediction experiments.
- `experiments/introduction/`: scripts for the introductory figure workflow.
- `utils/`: preprocessing, splitting, feature generation, and result aggregation
  utilities.

## Installation

Create a fresh environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric wheels depend on the local CUDA/Python version. If
the generic install command fails, install `torch` and `torch-geometric` from the
official instructions for your platform, then rerun the remaining requirements.

## Data and Paths

Most YAML files contain dataset, cache, output, and model paths. Before running
an experiment on a new machine, update the relevant paths in the selected config
file. Data directories and model/checkpoint directories may be stored together
locally after downloading the external archive. This release uses the sibling
directory `../data_models_results/` as the staging directory for external data,
model checkpoints, and generated result tables before Zenodo deposition. Empty
`models/` directories are kept in the repository as path placeholders; the
actual model files are stored in the external archive at the same relative
paths. The paths in the YAML files must point to the actual local locations.
Expected graph samples generally use the triplet format:

```text
<graph_id>_edges.npz
<graph_id>_features.npy
<graph_id>_label.json
```

Generated logs, outputs, results, caches, data folders, checkpoints, and model
files are ignored by `.gitignore` and should not be uploaded to GitHub unless
they are intentionally curated as small examples. Large reproducibility assets
should be archived separately.

## Data Availability

The datasets, trained model checkpoints, generated result tables, and large
intermediate artifacts required to reproduce the main experiments will be
deposited on Zenodo upon publication. The Zenodo record will include the raw or
preprocessed graph datasets, generated remnant/component datasets, baseline
result tables, and any model checkpoints needed to reproduce the reported
figures. In the local release layout, these external assets are staged next to
the repository under `../data_models_results/` before deposition.

After publication, cite the archive as:

```text
Zenodo DOI: to be added after deposition
```

After downloading the archive, place or symlink the data, model/checkpoint, and
result directories according to the paths used in the YAML files under
`configs/` and `experiments/*/configs/`, or update those YAML paths to match
your local directory layout.

## Reproducing Experiments

The complete command list is maintained in `run.sh`. Use that script as the
authoritative reproduction entry point rather than copying commands from several
README sections.

Run from the repository root:

```bash
chmod +x run.sh
./run.sh
```

Most commands in `run.sh` are commented because many steps are expensive or only
need to be run once. Uncomment the relevant block after the Zenodo data archive
has been downloaded and the paths have been configured.

The script is organized as follows:

```text
STEP 0  Data preparation
        Consolidate heuristic results, generate observation labels, and
        aggregate remnant/component datasets.

STEP 1  Model training
        Train baseline-comparison, trajectory-analysis, ablation, sensitivity,
        and early-warning model suites from YAML configs.

STEP 2  Introduction experiment, Fig. 1
        Generate example networks, run targeted attacks, compute metrics, and
        call experiments/introduction/plot1.py.

STEP 3  Baseline comparison, Fig. 3 plus ablation and sensitivity analyses
        Evaluate model checkpoints, compare against baselines, run exact and
        observation-label comparisons, and call experiments/baseline_comparison/plot3.py.

STEP 4  Trajectory analysis, Fig. 4 plus ablation and sensitivity analyses
        Run trajectory-property tests and call plot4.py and plot-sensitivity.py.

STEP 5  Early-warning signal and metric comparison, Fig. 6 and Fig. 7
        Run early-warning property tests, decision-window analysis, classical
        robustness-metric comparison, and call plot7.py.

STEP 6  Collapse-profile experiments
        Run collapse-profile evaluation and plotting.
```

Representative commands from `run.sh` include:

```bash
# Model training
python run_experiments.py --config configs/train-multisource.yaml
python run_experiments.py --config configs/train-multisource-exact.yaml
python run_experiments.py --config configs/train-ablation.yaml
python run_experiments.py --config configs/train-sensitivity.yaml

# Fig. 1
python experiments/introduction/run_experiment.py metrics
python experiments/introduction/plot1.py --net1 net1 --net2 net2 --tau_low 0.2 --tau_high 0.5 --row4_steps 5 6 10 11 --row4_attack BC --input_csv metrics.csv --opt_highlight_steps 5 10

# Fig. 3
python experiments/baseline_comparison/test_performance.py --config experiments/baseline_comparison/configs/test_multisource-LBWE-CPU.yaml
python experiments/baseline_comparison/test_performance.py --config experiments/baseline_comparison/configs/test_multisource-LBWE-GPU.yaml
python experiments/baseline_comparison/test_performance.py --config experiments/baseline_comparison/configs/test_multisource-REDDIT-CPU.yaml
python experiments/baseline_comparison/test_performance.py --config experiments/baseline_comparison/configs/test_multisource-REDDIT-GPU.yaml
python experiments/baseline_comparison/plot3.py

# Fig. 4
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-BA100.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-LFR100.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-london.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-power.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-route.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-ER2000.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-WS2000.yaml
python experiments/trajectory_analysis/plot4.py

# Fig. 6 and Fig. 7
python experiments/early_warning/test_properties.py --config experiments/early_warning/configs/london-111.yaml
python experiments/early_warning/test_properties.py --config experiments/early_warning/configs/london-185.yaml
python experiments/early_warning/test_properties.py --config experiments/early_warning/configs/power-353.yaml
python experiments/early_warning/test_properties.py --config experiments/early_warning/configs/power-588.yaml
python experiments/early_warning/plot7.py

# Collapse-profile analysis
python experiments/collapse_profile/test_profile.py --config experiments/collapse_profile/configs/test/test_reddit_profile.yaml
python experiments/collapse_profile/plot_profile.py --config experiments/collapse_profile/configs/plot/plot_reddit_profile.yaml
```

Each script contains a header with more specific inputs, outputs, and usage.

## Reproducibility Notes

- Set `seed` in YAML files for deterministic training where supported.
- GPU determinism can still vary across CUDA/cuDNN versions.
- Large raw datasets and trained checkpoints are not included by default. For a
  public release, provide a separate data-availability statement and stable links
  for any datasets or pretrained checkpoints required to reproduce the figures.

## License

This project is released under the MIT License. See `LICENSE` for details.