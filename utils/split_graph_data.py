"""
split_graph_data.py - Split Graph Dataset by Node Count

Description:
    Splits a folder of graph data files into subsets based on each graph's
    node count. Reads the node count from each *_label.json file, groups
    graphs into predefined node-count bins, and copies the corresponding
    file triplets (_edges.npz, _features.npy, _label.json) into separate
    output directories.

    The default bins and node-count range (10-40) are designed for a
    specific use case; adjust the split definitions as needed.

Input:
    A directory containing graph files with the naming convention:
        <prefix>_edges.npz
        <prefix>_features.npy
        <prefix>_label.json

Output:
    Subdirectories alongside the input folder, one per split bin:
        <parent>/<input_folder_name>_20_25_1/
        <parent>/<input_folder_name>_26_30_1/
        <parent>/<input_folder_name>_31_35_1/
        <parent>/<input_folder_name>_31_35_2/
        <parent>/<input_folder_name>_36_40_1/

Usage:
    python split_graph_data.py

    Edit the __main__ block to specify the target input directories,
    or import and call split_graph_data_by_node_count() directly.
"""

import os
import json
import shutil
import random
from collections import defaultdict
from tqdm import tqdm


# ---------------------------------------------------------------------------
#  Core splitting logic
# ---------------------------------------------------------------------------

def split_graph_data_by_node_count(input_folder, output_base_folder=None):
    """
    Split graph files in *input_folder* into sub-folders by node count.

    Each graph is identified by its file-name prefix (e.g. "LFR-20-40_0").
    Node counts are read from the corresponding *_label.json files.
    """
    if output_base_folder is None:
        output_base_folder = os.path.dirname(input_folder.rstrip('/'))

    input_folder_name = os.path.basename(input_folder.rstrip('/'))

    print(f"Scanning folder: {input_folder} ...")

    # ---- Discover label files ----

    node_count_to_files = defaultdict(list)

    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        print(f"Error: folder '{input_folder}' not found.")
        return

    label_suffix = '_label.json'
    json_files = [f for f in files if f.endswith(label_suffix)]

    print(f"Found {len(json_files)} label files. Reading node counts...")

    for json_file in tqdm(json_files, desc="Reading metadata"):
        json_path = os.path.join(input_folder, json_file)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            num_nodes = data.get('num_nodes')

            if num_nodes is not None and 10 <= num_nodes <= 40:
                prefix = json_file[:-len(label_suffix)]
                node_count_to_files[num_nodes].append(prefix)

        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # ---- Define splits ----

    splits = {
        '20_25_1': [],
        '26_30_1': [],
        '31_35_1': [],
        '31_35_2': [],
        '36_40_1': []
    }

    graphs_31_35 = []
    for nodes in range(31, 36):
        graphs_31_35.extend(node_count_to_files[nodes])

    random.shuffle(graphs_31_35)
    mid_point = len(graphs_31_35) // 2
    splits['31_35_1'] = graphs_31_35[:mid_point]
    splits['31_35_2'] = graphs_31_35[mid_point:]

    for nodes in range(10, 26):
        splits['20_25_1'].extend(node_count_to_files[nodes])

    for nodes in range(26, 31):
        splits['26_30_1'].extend(node_count_to_files[nodes])

    for nodes in range(36, 41):
        splits['36_40_1'].extend(node_count_to_files[nodes])

    # ---- Copy files ----

    print("\nCopying files...")

    for split_name, file_prefixes in splits.items():
        if not file_prefixes:
            continue

        output_folder = os.path.join(output_base_folder, f"{input_folder_name}_{split_name}")
        os.makedirs(output_folder, exist_ok=True)

        for prefix in tqdm(file_prefixes, desc=f"Copying to {split_name}"):
            files_to_copy = [
                f"{prefix}_edges.npz",
                f"{prefix}_features.npy",
                f"{prefix}_label.json"
            ]

            for file_name in files_to_copy:
                src_path = os.path.join(input_folder, file_name)
                dst_path = os.path.join(output_folder, file_name)

                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"Warning: file missing {src_path}")

    # ---- Summary ----

    print("\n" + "=" * 30)
    print(f"Done. Source folder: {input_folder_name}")
    total_copied = 0
    for split_name, file_prefixes in splits.items():
        count = len(file_prefixes)
        total_copied += count
        print(f"  - {split_name}: {count} graphs")
    print(f"Total copied: {total_copied} graphs")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)

    split_graph_data_by_node_count("./data_synth/exact-20-40/BA-exact-20-40")
    split_graph_data_by_node_count("./data_synth/exact-20-40/ER-exact-20-40")
    split_graph_data_by_node_count("./data_synth/exact-20-40/WS-exact-20-40")
