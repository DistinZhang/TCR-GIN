"""
TCR-GIN/utils/split.py - Split and Reorganize Graph Data by Node Count

Description:
    Reads graph data files from a source directory, determines the actual
    node count of each graph by inspecting its edge list, and moves the
    file triplet (_edges.npz, _features.npy, _label.json) into a new
    directory named after the corresponding 100-node bin.

    For example, a graph with 356 nodes originally in "data_synth/300-500/"
    will be moved to "data_synth/300/BA-300/" and renamed accordingly.

Input:
    A root folder containing graph files with the naming convention:
        <prefix>-<range>_<index>_edges.npz
        <prefix>-<range>_<index>_features.npy
        <prefix>-<range>_<index>_label.json
    e.g. BA-300-500_0_edges.npz

Output:
    Reorganized files under sibling directories of the root folder:
        <parent>/<bin>/<prefix>-<bin>/
    e.g. data_synth/300/BA-300/BA-300_0_edges.npz

    The original files are moved (not copied), so the source directory
    will be empty after processing and can be removed manually.

Usage:
    python split.py

    Edit the __main__ block to specify the target input directory,
    or import and call process_and_split_data() directly.
"""

import os
import re
import shutil
import numpy as np


# ---------------------------------------------------------------------------
#  Core splitting logic
# ---------------------------------------------------------------------------

def process_and_split_data(root_folder: str):
    """
    Move and rename graph data files into per-bin directories based on
    actual node count.

    :param root_folder: Path to the folder containing graph data files,
                        e.g. 'data_synth/300-500'.
    """
    if not os.path.isdir(root_folder):
        print(f"Error: folder '{root_folder}' does not exist.")
        return

    print(f"Processing folder: {root_folder}")

    parent_dir = os.path.dirname(root_folder)
    if parent_dir == "":
        parent_dir = "."

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.endswith("_edges.npz"):
                continue

            match = re.match(r'^(.*?)-(\d+-\d+)_(\d+)_edges\.npz$', filename)
            if not match:
                print(f"Warning: skipping file with unexpected name format: "
                      f"{os.path.join(dirpath, filename)}")
                continue

            prefix = match.group(1)
            original_range = match.group(2)
            index = match.group(3)

            base_name_src = f"{prefix}-{original_range}_{index}"
            src_edges_path = os.path.join(dirpath, f"{base_name_src}_edges.npz")
            src_features_path = os.path.join(dirpath, f"{base_name_src}_features.npy")
            src_label_path = os.path.join(dirpath, f"{base_name_src}_label.json")

            if not (os.path.exists(src_features_path) and os.path.exists(src_label_path)):
                print(f"Warning: skipping {src_edges_path} (missing features or label file).")
                continue

            try:
                with np.load(src_edges_path) as data:
                    edges = data['edges']
                    num_nodes = int(edges.max()) + 1
            except (KeyError, FileNotFoundError) as e:
                print(f"Error: failed to read {src_edges_path}: {e}. Skipping.")
                continue

            lower_bound = (num_nodes // 100) * 100

            new_top_level_dir = str(lower_bound)
            new_sub_dir = f"{prefix}-{lower_bound}"
            dest_dir = os.path.join(parent_dir, new_top_level_dir, new_sub_dir)
            os.makedirs(dest_dir, exist_ok=True)

            base_name_dest = f"{prefix}-{lower_bound}_{index}"
            dest_edges_path = os.path.join(dest_dir, f"{base_name_dest}_edges.npz")
            dest_features_path = os.path.join(dest_dir, f"{base_name_dest}_features.npy")
            dest_label_path = os.path.join(dest_dir, f"{base_name_dest}_label.json")

            print(f"Graph {base_name_src} has {num_nodes} nodes -> bin {lower_bound}-{lower_bound + 99}")
            try:
                shutil.move(src_edges_path, dest_edges_path)
                shutil.move(src_features_path, dest_features_path)
                shutil.move(src_label_path, dest_label_path)
                print(f"  - Moved to: {dest_dir}")
            except FileNotFoundError as e:
                print(f"Error: failed to move {base_name_src}: {e}")

    print("\nAll files processed.")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    input_folder_path = "data_synth/100"
    process_and_split_data(input_folder_path)
