"""
TCR-GIN/utils/split_train.py - Train / Validation / Test Split for Graph Datasets

Description:
    Splits graph network data into train, validation, and test sets
    (default ratio 80:10:10). Each network is identified by its file-name
    prefix and is expected to consist of three files:
        <prefix>_edges.npz
        <prefix>_features.npy
        <prefix>_label.json
    Only networks possessing all three files are considered valid.

    Multiple input directories can be supplied (comma-separated). Each
    directory is split independently, and the results are written into
    per-directory sub-folders under the output root.

Input (command-line arguments):
    --input_dirs   Comma-separated list of input directories containing
                   graph data files.
    --output / -o  Root output directory.
    --seed         Random seed (default: 42).

Output:
    <output>/<dir_name>/train/   Training set files
    <output>/<dir_name>/valid/   Validation set files
    <output>/<dir_name>/test/    Test set files

Usage:
    python split_train.py \
        --input_dirs "data_synth/100/BA-100,data_synth/100/ER-100,data_synth/100/WS-100,data_synth/100/LFR-100" \
        --output split/100

    python split_train.py \
        --input_dirs "data_real/power/datasets/power-300" \
        --output split/300
"""

import os
import random
import shutil
import re
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
#  Data collection
# ---------------------------------------------------------------------------

def collect_network_data(input_dirs):
    """Scan all input directories and return valid networks (those with all three file types)."""
    all_networks = defaultdict(lambda: {'dirs': set(), 'files': []})

    for input_dir in input_dirs:
        folder_name = os.path.basename(input_dir)
        if not os.path.exists(input_dir):
            print(f"Warning: directory '{input_dir}' does not exist, skipped.")
            continue

        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if not os.path.isfile(file_path):
                continue

            match = re.match(r'(.+?)_(edges\.npz|features\.npy|label\.json)$', filename)
            if match:
                network_name = match.group(1)
                all_networks[network_name]['dirs'].add(folder_name)
                all_networks[network_name]['files'].append((file_path, filename))

    valid_networks = {}
    for network_name, data in all_networks.items():
        file_extensions = set()
        for _, filename in data['files']:
            if filename.endswith('edges.npz'):
                file_extensions.add('npz')
            elif filename.endswith('features.npy'):
                file_extensions.add('npy')
            elif filename.endswith('label.json'):
                file_extensions.add('json')

        if len(file_extensions) == 3:
            valid_networks[network_name] = data

    return valid_networks


# ---------------------------------------------------------------------------
#  Train / valid / test splitting
# ---------------------------------------------------------------------------

def split_data(valid_networks, output_dir):
    """Split networks per directory into train/valid/test and copy files."""
    networks_by_dir = defaultdict(list)
    for network_name, data in valid_networks.items():
        for dir_name in data['dirs']:
            networks_by_dir[dir_name].append(network_name)

    for dir_name in networks_by_dir:
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(output_dir, dir_name, split), exist_ok=True)

    for dir_name, networks in networks_by_dir.items():
        random.shuffle(networks)
        total = len(networks)

        train_ratio, val_ratio = 0.8, 0.1
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_networks = networks[:train_end]
        val_networks = networks[train_end:val_end]
        test_networks = networks[val_end:]

        for network_name in train_networks:
            dest_dir = os.path.join(output_dir, dir_name, 'train')
            copy_network_files(valid_networks[network_name], dir_name, dest_dir)

        for network_name in val_networks:
            dest_dir = os.path.join(output_dir, dir_name, 'valid')
            copy_network_files(valid_networks[network_name], dir_name, dest_dir)

        for network_name in test_networks:
            dest_dir = os.path.join(output_dir, dir_name, 'test')
            copy_network_files(valid_networks[network_name], dir_name, dest_dir)

        print(f"Directory {dir_name} split complete: "
              f"train {len(train_networks)}, valid {len(val_networks)}, test {len(test_networks)}")


# ---------------------------------------------------------------------------
#  File copy helper
# ---------------------------------------------------------------------------

def copy_network_files(network_data, dir_name, target_dir):
    """Copy all files belonging to a single network into *target_dir*."""
    for src_path, filename in network_data['files']:
        file_dir_name = os.path.basename(os.path.dirname(src_path))
        if file_dir_name == dir_name:
            dst_path = os.path.join(target_dir, filename)
            shutil.copy2(src_path, dst_path)


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split graph network data into train / validation / test sets (80:10:10)."
    )
    parser.add_argument('--input_dirs', required=True,
                        help='Comma-separated input directory paths, e.g. "path1,path2,path3"')
    parser.add_argument('--output', '-o', required=True,
                        help='Root output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()
    random.seed(args.seed)

    input_dirs = [path.strip().replace('\\', '/') for path in args.input_dirs.split(',')]
    output_dir = args.output.replace('\\', '/')

    print("Collecting network data...")
    valid_networks = collect_network_data(input_dirs)
    print(f"Found {len(valid_networks)} valid networks.")

    print("Splitting data...")
    split_data(valid_networks, output_dir)

    print(f"Done. Results saved to '{output_dir}'.")


if __name__ == '__main__':
    main()
