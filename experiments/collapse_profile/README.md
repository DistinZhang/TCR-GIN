# collapse_profile

This branch keeps the original TCR-GIN backbone and switches the task from
single-threshold scalar regression to multi-threshold collapse-profile
regression.

## Expected profile label file

Store profile labels separately from the old `*_label.json` files:

- old scalar label: `graph_xxx_label.json`
- new profile label: `graph_xxx_profile_label.json`

Minimum required keys in each profile label file:

- `tau_grid_full`
- `collapse_profile_full`

## Main scripts

- `scripts/train_profile.py`: profile training entry point
- `scripts/data_loader_profile.py`: reads profile labels and slices the active tau grid
- `model/tcr_gin_profile.py`: TCR-GIN backbone with vector output head
- `scripts/run_profile_experiments.py`: runs one config or all configs in a directory

## One-time directory bootstrap

```bash
bash experiments/collapse_profile/scripts/bootstrap_dirs.sh
```

## Run a single config

```bash
python experiments/collapse_profile/scripts/train_profile.py \
  --config experiments/collapse_profile/configs/transport_profile.yaml
```

## Run all configs in the folder

```bash
python experiments/collapse_profile/scripts/run_profile_experiments.py \
  --config_dir experiments/collapse_profile/configs
```
