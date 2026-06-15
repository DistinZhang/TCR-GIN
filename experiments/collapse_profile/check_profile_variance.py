"""
check_profile_variance.py
Check variance distribution of collapse profile dataset, diagnose if model faces "no learnable signal" problem.

Usage:
  python check_profile_variance.py \
    --dirs /root/autodl-tmp/ERGM/split-ergm/REDDIT/configuration/train \
           /root/autodl-tmp/ERGM/split-ergm/REDDIT/erdos_renyi/train \
    --label_suffix _profile_label.json \
    --profile_key collapse_profile_full \
    --tau_key tau_grid_full
    python experiments/collapse_profile/check_profile_variance.py --dirs /root/autodl-tmp/ERGM/split-ergm/REDDIT/configuration/train /root/autodl-tmp/ERGM/split-ergm/REDDIT/erdos_renyi/train --output_dir experiments/collapse_profile/variance_check
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Headless server environment, disable GUI backend
import matplotlib.pyplot as plt


# ─────────────────────────── Arguments ───────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs", nargs="+", required=True,
        help="One or more directories containing label JSON files"
    )
    parser.add_argument(
        "--label_suffix", default="_profile_label.json",
        help="Label file suffix"
    )
    parser.add_argument(
        "--profile_key", default="collapse_profile_full",
        help="Key name of the profile array in JSON"
    )
    parser.add_argument(
        "--tau_key", default="tau_grid_full",
        help="Key name of the tau grid in JSON"
    )
    parser.add_argument(
        "--output_dir", default=".",
        help="Directory to save figures"
    )
    return parser.parse_args()


# ─────────────────────────── source name generation ───────────────────────────

def make_source_name(d: str, existing: set) -> str:
    """
    Use "grandparent_parent_cur" combination to construct unique names, avoid multi-directory name conflicts.
    Example:/root/.../REDDIT/configuration/train -> configuration_train
          /root/.../REDDIT/erdos_renyi/train   -> erdos_renyi_train
    """
    parts = []
    p = os.path.normpath(d)
    for _ in range(3):          # Take at most 3 levels from the end
        parts.insert(0, os.path.basename(p))
        p = os.path.dirname(p)

    # Start from the shortest uniquely distinguishable combination
    for depth in range(1, len(parts) + 1):
        name = "_".join(parts[-depth:])
        if name not in existing:
            return name

    # If still duplicate, add numeric suffix
    base = "_".join(parts)
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


# ─────────────────────────── Data Loading ───────────────────────────

def load_profiles(dirs, label_suffix, profile_key, tau_key):
    """
    Returns:
        data_by_source: dict[source_name -> {"profiles": np.ndarray [N, T], "taus": list}]
    """
    data_by_source = {}
    used_names = set()

    for d in dirs:
        source_name = make_source_name(d, used_names)
        used_names.add(source_name)

        profiles = []
        taus = None

        if not os.path.isdir(d):
            print(f"[WARNING] Directory not found: {d}, skipped.")
            continue

        files = sorted(f for f in os.listdir(d) if f.endswith(label_suffix))
        if not files:
            print(f"[WARNING] No {label_suffix} files found in {d}, skipped.")
            continue

        for fname in files:
            fpath = os.path.join(d, fname)
            with open(fpath, "r") as fp:
                obj = json.load(fp)

            profile = obj.get(profile_key)
            if profile is None:
                continue
            profiles.append(profile)

            if taus is None and tau_key in obj:
                taus = obj[tau_key]

        arr = np.array(profiles, dtype=np.float32)   # [N, T]
        data_by_source[source_name] = {"profiles": arr, "taus": taus, "path": d}
        print(f"[OK] {source_name}: {len(arr)} graphs, {arr.shape[1]} tau points  ({d})")

    return data_by_source


# ─────────────────────────── Statistical Analysis ───────────────────────────

def compute_stats(arr: np.ndarray, taus):
    T = arr.shape[1]
    tau_labels = taus if taus else [f"tau_{i}" for i in range(T)]

    stats = {
        "mean_per_tau":        arr.mean(axis=0),
        "std_per_tau":         arr.std(axis=0),
        "min_per_tau":         arr.min(axis=0),
        "max_per_tau":         arr.max(axis=0),
        "range_per_tau":       arr.max(axis=0) - arr.min(axis=0),
        "cv_per_tau":          arr.std(axis=0) / (arr.mean(axis=0) + 1e-8),
        "global_std":          arr.std(),
        "global_mean":         arr.mean(),
        "n_graphs":            arr.shape[0],
        "tau_labels":          tau_labels,
        "per_graph_mean_std":  arr.mean(axis=1).std(),   # cross-graph spread
    }
    return stats


def print_stats_table(name, stats):
    print(f"\n{'='*65}")
    print(f"  Source: {name}  ({stats['n_graphs']} graphs)")
    print(f"{'='*65}")
    print(f"  Global mean : {stats['global_mean']:.6f}")
    print(f"  Global std  : {stats['global_std']:.6f}")
    print(f"  Cross-graph mean std : {stats['per_graph_mean_std']:.6f}")
    print()
    print(f"  {'tau':>8}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}  {'CV':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for i, tau in enumerate(stats["tau_labels"]):
        print(
            f"  {str(tau):>8}"
            f"  {stats['mean_per_tau'][i]:>10.6f}"
            f"  {stats['std_per_tau'][i]:>10.6f}"
            f"  {stats['min_per_tau'][i]:>10.6f}"
            f"  {stats['max_per_tau'][i]:>10.6f}"
            f"  {stats['cv_per_tau'][i]:>8.3f}"
        )

    low_std_taus = [
        str(stats["tau_labels"][i])
        for i, s in enumerate(stats["std_per_tau"])
        if s < 0.005
    ]
    if low_std_taus:
        print(f"\n  [!] tau positions with std < 0.005 (weak signal): {low_std_taus}")
    else:
        print(f"\n  [OK] All tau positions have std >= 0.005. Data has learnable signal.")


# ─────────────────────────── Visualization ───────────────────────────

def plot_profile_distribution(name, arr, stats, output_dir):
    taus = stats["tau_labels"]
    T = len(taus)
    x = np.arange(T)
    tau_str = [str(t) for t in taus]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Collapse Profile Variance Analysis - {name}", fontsize=13)

    # ── Panel 1: mean +/- std ──
    ax = axes[0]
    mean = stats["mean_per_tau"]
    std  = stats["std_per_tau"]
    ax.plot(x, mean, "b-o", label="mean", linewidth=2, markersize=4)
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, label="+/- 1 std")
    ax.fill_between(x, stats["min_per_tau"], stats["max_per_tau"],
                    alpha=0.1, color="gray", label="[min, max]")
    ax.set_xticks(x)
    ax.set_xticklabels(tau_str, rotation=45, fontsize=8)
    ax.set_title("Mean +/- Std (per tau)")
    ax.set_xlabel("tau")
    ax.set_ylabel("Collapse Fraction")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ── Panel 2: std bar chart ──
    ax = axes[1]
    bars = ax.bar(x, std, color="steelblue", alpha=0.8)
    ax.axhline(0.005, color="red", linestyle="--", linewidth=1.2,
               label="std=0.005 warning threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(tau_str, rotation=45, fontsize=8)
    ax.set_title("Std per tau (higher = more learnable)")
    ax.set_xlabel("tau")
    ax.set_ylabel("std")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, std):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(std) * 0.01,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=6
        )

    # ── Panel 3: 50 random profile overlay ──
    ax = axes[2]
    sample_idx = np.random.choice(len(arr), size=min(50, len(arr)), replace=False)
    for idx in sample_idx:
        ax.plot(x, arr[idx], color="steelblue", alpha=0.2, linewidth=0.8)
    ax.plot(x, mean, "r-", linewidth=2, label="mean profile")
    ax.set_xticks(x)
    ax.set_xticklabels(tau_str, rotation=45, fontsize=8)
    ax.set_title("Random 50 profiles overlay")
    ax.set_xlabel("tau")
    ax.set_ylabel("Collapse Fraction")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    # Remove possible special characters from filenames
    safe_name = name.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(output_dir, f"profile_variance_{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {out_path}")


def plot_cross_source_comparison(all_stats, output_dir):
    if len(all_stats) < 2:
        return

    # Take tau labels from first source as x-axis (assume same for all sources)
    first_stats = next(iter(all_stats.values()))
    taus  = first_stats["tau_labels"]
    T     = len(taus)
    x     = np.arange(T)
    tau_str = [str(t) for t in taus]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-source Collapse Profile Comparison", fontsize=13)
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_stats)))

    # ── Panel 1: std comparison ──
    ax = axes[0]
    for (name, stats), color in zip(all_stats.items(), colors):
        ax.plot(x, stats["std_per_tau"], "-o", label=name,
                color=color, linewidth=2, markersize=4)
    ax.axhline(0.005, color="red", linestyle="--", linewidth=1.2,
               label="std=0.005 warning")
    ax.set_xticks(x)
    ax.set_xticklabels(tau_str, rotation=45, fontsize=8)
    ax.set_title("Std per tau by source")
    ax.set_xlabel("tau")
    ax.set_ylabel("std")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # ── Panel 2: mean profile comparison ──
    ax = axes[1]
    for (name, stats), color in zip(all_stats.items(), colors):
        mean = stats["mean_per_tau"]
        std  = stats["std_per_tau"]
        ax.plot(x, mean, "-o", label=name, color=color, linewidth=2, markersize=4)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(tau_str, rotation=45, fontsize=8)
    ax.set_title("Mean profile by source (shading = +/- 1 std)")
    ax.set_xlabel("tau")
    ax.set_ylabel("Collapse Fraction")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "profile_variance_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Saved] {out_path}")


# ─────────────────────────── Main Process ───────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("  Collapse Profile Variance Inspector")
    print("=" * 65)

    data_by_source = load_profiles(
        args.dirs, args.label_suffix, args.profile_key, args.tau_key
    )

    if not data_by_source:
        print("[ERROR] No data loaded. Check --dirs and --label_suffix.")
        return

    all_stats = {}
    for name, data in data_by_source.items():
        arr   = data["profiles"]
        taus  = data["taus"]
        stats = compute_stats(arr, taus)
        print_stats_table(name, stats)
        plot_profile_distribution(name, arr, stats, args.output_dir)
        all_stats[name] = stats

    plot_cross_source_comparison(all_stats, args.output_dir)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    for name, stats in all_stats.items():
        min_std  = stats["std_per_tau"].min()
        mean_std = stats["std_per_tau"].mean()
        cv_mean  = stats["cv_per_tau"].mean()
        print(f"\n  [{name}]")
        print(f"    Min per-tau std  : {min_std:.6f}  {'[!] Very low' if min_std < 0.005 else '[OK]'}")
        print(f"    Mean per-tau std : {mean_std:.6f}")
        print(f"    Mean CV          : {cv_mean:.3f}  {'[!] Low' if cv_mean < 0.1 else '[OK]'}")
        if min_std < 0.005 or cv_mean < 0.1:
            print("    --> Diagnosis: Weak discriminative signal. Model may collapse to mean prediction.")
            print("        Suggestion: Check data generation params or add structural node features.")
        else:
            print("    --> Diagnosis: Data has sufficient variance. Learnable signal exists.")
            print("        Suggestion: Focus on improving model expressiveness (features/architecture).")


if __name__ == "__main__":
    main()
