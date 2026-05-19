#!/bin/bash
# =============================================================================
#  TCR-GIN/run.sh — Master experiment runner for TCR-GIN
#
#  Orchestrates all experiments: data preparation, training, evaluation,
#  and figure generation.  Commented-out commands are retained as one-time
#  or optional steps; uncomment as needed.
#
#  Usage:
#      chmod +x run.sh
#      ./run.sh
# =============================================================================

set -e

echo "================================================"
echo "  TCR-GIN — Batch Experiment Runner"
echo "================================================"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 0 — DATA PREPARATION  (one-time steps, uncomment when needed)       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 0a. Consolidate heuristic results & generate observation labels ──────────
#
python utils/network_data_consolidator.py -n ./autodl-tmp/split/100 -r ./autodl-tmp/data_synth/100/results -o ./autodl-tmp/data_synth/100/results_final
python utils/network_data_consolidator.py -n ./autodl-tmp/split/100/power-100 -r ./autodl-tmp/data_trajectory/power/datasets/100/results -o ./autodl-tmp/data_trajectory/power/datasets/100/results_final
python utils/network_data_consolidator.py -n ./autodl-tmp/data_trajectory/power/power-Components/power-Components -r ./autodl-tmp/data_trajectory/power/power-Components/results -o ./autodl-tmp/data_trajectory/power/power-Components/results_final

# ── 0b. Aggregate remnants — synthetic networks (100 nodes) ─────────────────
#
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/LFR100/LFR100-Components --remnants-dir /root/autodl-tmp/data_exploratory/LFR100/LFR100-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/WS100/WS100-Components --remnants-dir /root/autodl-tmp/data_exploratory/WS100/WS100-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/BA100/BA100-Components --remnants-dir /root/autodl-tmp/data_exploratory/BA100/BA100-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/ER100/ER100-Components --remnants-dir /root/autodl-tmp/data_exploratory/ER100/ER100-Remnants

# ── 0c. Aggregate remnants — synthetic networks (2000 nodes) ────────────────
#
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/BA2000/BA2000-Components --remnants-dir /root/autodl-tmp/data_exploratory/BA2000/BA2000-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/LFR2000/LFR2000-Components --remnants-dir /root/autodl-tmp/data_exploratory/LFR2000/LFR2000-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/ER2000/ER2000-Components --remnants-dir /root/autodl-tmp/data_exploratory/ER2000/ER2000-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/WS2000/WS2000-Components --remnants-dir /root/autodl-tmp/data_exploratory/WS2000/WS2000-Remnants

# ── 0d. Aggregate remnants — real-world networks ────────────────────────────
#
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/power/power-Components --remnants-dir /root/autodl-tmp/data_exploratory/power/power-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/london/london-Components --remnants-dir /root/autodl-tmp/data_exploratory/london/london-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_exploratory/route/route-Components --remnants-dir /root/autodl-tmp/data_exploratory/route/route-Remnants

# ── 0e. Aggregate remnants — early-warning metric datasets ──────────────────
#
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_metric/london_111/london_111-Components --remnants-dir /root/autodl-tmp/data_metric/london_111/london_111-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_metric/london_185/london_185-Components --remnants-dir /root/autodl-tmp/data_metric/london_185/london_185-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_metric/power_353/power_353-Components --remnants-dir /root/autodl-tmp/data_metric/power_353/power_353-Remnants
python utils/aggregate_remnants.py --components-dir /root/autodl-tmp/data_metric/power_588/power_588-Components --remnants-dir /root/autodl-tmp/data_metric/power_588/power_588-Remnants


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — MODEL TRAINING  (uncomment the configs you need)                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 1a. Baseline comparison training ────────────────────────────────────────
#
python run_experiments.py --config configs/train-multisource.yaml
python run_experiments.py --config configs/train-multisource-exact.yaml
python run_experiments.py --config configs/train-ablation.yaml
python run_experiments.py --config configs/train-sensitivity.yaml

# ── 1b. Trajectory analysis training ────────────────────────────────────────
#
python run_experiments.py --config ./experiments/trajectory_analysis/configs/train-multisource.yaml
python run_experiments.py --config ./experiments/trajectory_analysis/configs/train-ablation.yaml
python run_experiments.py --config ./experiments/trajectory_analysis/configs/train-sensitivity.yaml

# ── 1c. Early-warning experiment training (subgraph-specific models) ────────
#
python run_experiments.py --config ./experiments/trajectory_analysis/configs/train-multisource-111.yaml
python run_experiments.py --config ./experiments/trajectory_analysis/configs/train-multisource-185.yaml
python run_experiments.py --config ./experiments/trajectory_analysis/configs/train-multisource-353.yaml
python run_experiments.py --config ./experiments/trajectory_analysis/configs/train-multisource-588.yaml


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — INTRODUCTION EXPERIMENT  (Fig. 1)                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 2a. Generate example networks ───────────────────────────────────────────
#
python experiments/introduction/run_experiment.py generate --name net1 --type WS --n 20 --k 6 --p 0.1
python experiments/introduction/run_experiment.py generate --name net1 --type BA --n 30 --m 3
python experiments/introduction/run_experiment.py generate --name net1 --type ER --n 20 --avg_degree 4

# ── 2b. Run targeted attacks ────────────────────────────────────────────────
#
python experiments/introduction/run_experiment.py attack --tau 10 --nets net1 --force --workers 16
python experiments/introduction/run_experiment.py attack --tau 4 --nets net1 --force --workers 16
python experiments/introduction/run_experiment.py attack --tau 10 --nets net2 --force --workers 16
python experiments/introduction/run_experiment.py attack --tau 4 --nets net2 --force --workers 16

# ── 2c. Compute robustness metrics ─────────────────────────────────────────
#
python experiments/introduction/run_experiment.py metrics

# ── 2d. Generate Fig. 1 ────────────────────────────────────────────────────
#
python experiments/introduction/plot1.py --net1 net1 --net2 net2 --tau_low 0.2 --tau_high 0.5 --row4_steps 5 6 10 11 --row4_attack BC --input_csv metrics.csv --opt_highlight_steps 5 10


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — BASELINE COMPARISON  (Fig. 3, ablation, sensitivity)            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 3a. Performance evaluation (Fig. 3a–e) ──────────────────────────────────
#
python experiments/baseline_comparison/test_performance.py --config experiments/baseline_comparison/configs/test_multisource-LBWE-CPU.yaml
python experiments/baseline_comparison/test_performance.py --config experiments/baseline_comparison/configs/test_multisource-LBWE-GPU.yaml
python experiments/baseline_comparison/test_performance.py --config experiments/baseline_comparison/configs/test_multisource-REDDIT-CPU.yaml
python experiments/baseline_comparison/test_performance.py --config experiments/baseline_comparison/configs/test_multisource-REDDIT-GPU.yaml

# ── 3b. Real-world generalisation test ──────────────────────────────── 
#
python experiments/baseline_comparison/real_generalization/test_generalization_real.py --config experiments/baseline_comparison/real_generalization/configs/test_generalization_real.yaml

# ── 3c. Inference time benchmark (Fig. 3f) ─────────────────────────── 
#
python experiments/baseline_comparison/test_performance.py --config experiments/baseline_comparison/configs/test_multisource-LBWE-CPU-time.yaml

# ── 3d. Exact & observation-label comparison (Fig. 3g–h) ───────────── 
#
python experiments/baseline_comparison/exact_comparison/test_performance.py --config experiments/baseline_comparison/exact_comparison/configs/test_exact.yaml
python experiments/baseline_comparison/exact_comparison/test_performance.py --config experiments/baseline_comparison/exact_comparison/configs/test_observ.yaml

# ── 3e. Generate Fig. 3 ────────────────────────────────────────────── 
#
python experiments/baseline_comparison/plot3.py

# ── 3f. Ablation study ─────────────────────────────────────────────── 
#
python experiments/baseline_comparison/ablation/test_ablation.py --config experiments/baseline_comparison/ablation/configs/test_ablation_synth.yaml
python experiments/baseline_comparison/ablation/test_ablation.py --config experiments/baseline_comparison/ablation/configs/test_ablation_REDDIT.yaml

# ── 3g. Sensitivity analysis ───────────────────────────────────────── 
#
python experiments/baseline_comparison/sensitivity/test_sensitivity.py --config experiments/baseline_comparison/sensitivity/configs/test_sensitivity_synth.yaml
python experiments/baseline_comparison/sensitivity/test_sensitivity.py --config experiments/baseline_comparison/sensitivity/configs/test_sensitivity_REDDIT.yaml


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — TRAJECTORY ANALYSIS  (Fig. 4, ablation, sensitivity)            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 4a. Multi-source property tests (main results) ─────────────────── 
#
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-BA100.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-LFR100.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-london.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-power.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-route.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-ER2000.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/test_properties_base_multisource-WS2000.yaml

# ── 4b. Ablation property tests ────────────────────────────────────── 
# #
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/ablation/test_properties_ablation-BA100.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/ablation/test_properties_ablation-LFR100.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/ablation/test_properties_ablation-london.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/ablation/test_properties_ablation-power.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/ablation/test_properties_ablation-ER2000.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/ablation/test_properties_ablation-WS2000.yaml

# ── 4c. Sensitivity property tests ─────────────────────────────────── 
#
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/sensitivity/test_properties_sensitivity-BA100.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/sensitivity/test_properties_sensitivity-LFR100.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/sensitivity/test_properties_sensitivity-london.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/sensitivity/test_properties_sensitivity-power.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/sensitivity/test_properties_sensitivity-ER2000.yaml
python experiments/trajectory_analysis/test_properties.py --config experiments/trajectory_analysis/configs/sensitivity/test_properties_sensitivity-WS2000.yaml

# ── 4d. Generate Fig. 4 and sensitivity plots ──────────────────────── 
# #
python experiments/trajectory_analysis/plot4.py
python experiments/trajectory_analysis/plot-sensitivity.py


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — EARLY-WARNING SIGNAL & METRIC COMPARISON  (Fig. 6, Fig. 7)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 5a. Trajectory property tests ───────────────────────────────────────────
#
python experiments/early_warning/test_properties.py --config experiments/early_warning/configs/london-111.yaml
python experiments/early_warning/test_properties.py --config experiments/early_warning/configs/london-185.yaml
python experiments/early_warning/test_properties.py --config experiments/early_warning/configs/power-353.yaml
python experiments/early_warning/test_properties.py --config experiments/early_warning/configs/power-588.yaml

# ── 5b. Decision window analysis (Fig. 6, SI Fig. 12–15) ───────────────────
# #
python experiments/early_warning/calculate_decision_window.py --input_root "/root/autodl-tmp/data_metric/london_111" --output_dir "experiments/early_warning/results/decision_window/london-111" --initial_size 369 --collapse_target 0.30 --model_config "experiments/early_warning/configs/london-111.yaml"
python experiments/early_warning/calculate_decision_window.py --input_root "/root/autodl-tmp/data_metric/london_185" --output_dir "experiments/early_warning/results/decision_window/london-185" --initial_size 369 --collapse_target 0.50 --model_config "experiments/early_warning/configs/london-185.yaml"
python experiments/early_warning/calculate_decision_window.py --input_root "/root/autodl-tmp/data_metric/power_353" --output_dir "experiments/early_warning/results/decision_window/power-353" --initial_size 1176 --collapse_target 0.30 --model_config "experiments/early_warning/configs/power-353.yaml"
python experiments/early_warning/calculate_decision_window.py --input_root "/root/autodl-tmp/data_metric/power_588" --output_dir "experiments/early_warning/results/decision_window/power-588" --initial_size 1176 --collapse_target 0.50 --model_config "experiments/early_warning/configs/power-588.yaml"

# ── 5c. Classic robustness metric comparison (SI Fig. 16–19) ────────────────
#
python experiments/early_warning/evaluate_classic_robustness.py --input_root "/root/autodl-tmp/data_metric/london_111" --output_dir "experiments/early_warning/results/robustness_metrics/london-111" --initial_size 369 --collapse_target 0.30 --model_config "experiments/early_warning/configs/london-111.yaml"
python experiments/early_warning/evaluate_classic_robustness.py --input_root "/root/autodl-tmp/data_metric/london_185" --output_dir "experiments/early_warning/results/robustness_metrics/london-185" --initial_size 369 --collapse_target 0.50 --model_config "experiments/early_warning/configs/london-185.yaml"
python experiments/early_warning/evaluate_classic_robustness.py --input_root "/root/autodl-tmp/data_metric/power_353" --output_dir "experiments/early_warning/results/robustness_metrics/Power-353" --initial_size 1176 --collapse_target 0.30 --model_config "experiments/early_warning/configs/power-353.yaml"
python experiments/early_warning/evaluate_classic_robustness.py --input_root "/root/autodl-tmp/data_metric/power_588" --output_dir "experiments/early_warning/results/robustness_metrics/Power-588" --initial_size 1176 --collapse_target 0.50 --model_config "experiments/early_warning/configs/power-588.yaml"

# ── 5d. Generate Fig. 7 ────────────────────────────────────────────────────
#
python experiments/early_warning/plot7.py


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 6 — collapse_profile                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

python experiments/collapse_profile/run_profile_experiments.py --config experiments/collapse_profile/configs/train/reddit_profile-lstm.yaml
python experiments/collapse_profile/test_profile.py --config experiments/collapse_profile/configs/test/test_reddit_profile.yaml
python experiments/collapse_profile/plot_profile.py --config experiments/collapse_profile/configs/plot/plot_reddit_profile.yaml
