# REDDIT Collapse-Profile Experiment

This directory contains the REDDIT collapse-profile extension of TCR-GIN. The
experiment keeps the original TCR-GIN backbone and changes the task from scalar
collapse-distance regression to vector-valued collapse-profile regression over a
discrete tau grid.

## Retained contents

```text
collapse_profile/
├── arguments_profile.py
├── data_loader_profile.py
├── train_profile.py
├── test_profile.py
├── plot_profile.py
├── run_profile_experiments.py
├── utils_profile.py
├── model/tcr_gin_profile.py
├── configs/train/reddit_profile.yaml
├── configs/test/test_reddit_profile.yaml
├── configs/plot/plot_reddit_profile.yaml
├── models/reddit_profile/model_run_*.pt
└── results/reddit_profile/
```

One-off diagnostic scripts, feature-repair scripts, notebook checkpoints,
Python bytecode caches, and training logs were removed from the archive.

## Expected profile label file

Profile labels are stored separately from scalar `*_label.json` files:

- scalar label: `graph_xxx_label.json`
- profile label: `graph_xxx_profile_label.json`

Minimum required keys in each profile label file:

- `tau_grid_full`
- `collapse_profile_full`

## Data paths

The YAML files currently use the original server layout under
`/root/autodl-tmp/profile/...`. If the Zenodo archive is unpacked elsewhere,
update these paths to the corresponding directories under `data/profile/`.

## Run commands

Train the five REDDIT profile runs:

```bash
python experiments/collapse_profile/run_profile_experiments.py \
  --config experiments/collapse_profile/configs/train/reddit_profile.yaml
```

Evaluate the retained checkpoints:

```bash
python experiments/collapse_profile/test_profile.py \
  --config experiments/collapse_profile/configs/test/test_reddit_profile.yaml
```

Generate the profile plots:

```bash
python experiments/collapse_profile/plot_profile.py \
  --config experiments/collapse_profile/configs/plot/plot_reddit_profile.yaml
```
