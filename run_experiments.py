#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCR-GIN/run_experiments.py

Run experiment campaigns from a YAML configuration file, including batch,
multi-source, dynamic-source, grid-search, ablation, and sensitivity workflows.

Function
--------
This script:
1. Loads a campaign YAML configuration
2. Builds experiment tasks from:
   - base configuration
   - dataset instances
   - parameter search combinations
3. Supports dataset preparation modes:
   - batch
   - multi_source
   - dynamic_source
4. Launches `train_piss.py` for each generated task
5. Saves stdout/stderr logs for each run
6. Cleans temporary data and cache directories after execution
7. Parses experiment logs and exports a summarized Excel report

Inputs
------
- `--config`: path to a YAML experiment configuration file

Outputs
-------
- Per-run log files in the configured output directory
- Model checkpoints in the configured model directory
- `summary_report_<campaign_name>.xlsx` under the campaign output directory

Usage
-----
Example:
    python run_experiments.py --config configs/train-base.yaml
    python run_experiments.py --config configs/train-multisource.yaml
    python run_experiments.py --config configs/train-ablation.yaml
    python run_experiments.py --config configs/train-sensitivity.yaml
"""

import os
import re
import copy
import json
import yaml
import shutil
import argparse
import itertools
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ==============================================================================
# Constants
# ==============================================================================
PYTHON_EXECUTABLE = "python"
TRAIN_SCRIPT = "train_piss.py"


# ==============================================================================
# Multi-source dataset preparation and cleanup
# ==============================================================================
def setup_symlinked_dirs(config):
    """Create temporary aggregated directories for multi-source datasets."""
    temp_dir = Path(config['temp_data_dir'])
    print(f"[*] Preparing multi-source aggregation directory: {temp_dir.resolve()}")

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    symlinked_paths = {
        'train': temp_dir / 'train',
        'val': temp_dir / 'valid',
        'test': temp_dir / 'test'
    }

    for path in symlinked_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    source_map = {
        'train': config['train_source_dirs'],
        'val': config['val_source_dirs'],
        'test': config['test_source_dirs']
    }

    for split, source_dirs in source_map.items():
        dest_dir = symlinked_paths[split]
        for source_dir_str in source_dirs:
            source_dir = Path(source_dir_str)
            if not source_dir.is_dir():
                continue
            for src_file in source_dir.iterdir():
                dest_file = dest_dir / src_file.name
                if not dest_file.exists():
                    os.symlink(src_file.resolve(), dest_file)

    return {
        'train_path': str(symlinked_paths['train']),
        'val_path': str(symlinked_paths['val']),
        'test_path': str(symlinked_paths['test'])
    }


def cleanup_symlinked_dirs(config):
    """Remove temporary directories created for multi-source datasets."""
    temp_dir = Path(config['temp_data_dir'])
    if temp_dir.exists():
        print(f"[*] Cleaning temporary aggregation directory: {temp_dir.resolve()}")
        shutil.rmtree(temp_dir)


# ==============================================================================
# Dynamic-source dataset preparation and cleanup
# ==============================================================================
def prepare_dynamic_source_dirs(instance_config, master_root, templates):
    """
    Filter and copy files from a master dataset source into a local temporary
    directory according to a dynamic-source configuration.
    """
    mix_id = instance_config['mix_id']
    temp_data_dir_str = templates['temp_data_dir'].format(mix_id=mix_id)
    local_temp_root = Path(temp_data_dir_str)

    local_paths = {
        'train': local_temp_root / 'train',
        'val': local_temp_root / 'valid',
        'test': local_temp_root / 'test'
    }

    if local_temp_root.exists() and any((local_temp_root / 'train').iterdir()):
        print(f"[+] Existing dynamic dataset '{mix_id}' was found. Skipping file copy.")
        return {
            'train_path': str(local_paths['train']),
            'val_path': str(local_paths['val']),
            'test_path': str(local_paths['test']),
            '_temp_dir_to_cleanup': str(local_temp_root)
        }

    print(f"[*] Creating dynamic dataset '{mix_id}' at: {local_temp_root.resolve()}")

    if local_temp_root.exists():
        shutil.rmtree(local_temp_root)

    for path in local_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    files_to_copy = []

    for source_group in instance_config['sources']:
        for base_dir in source_group['base_dirs']:
            for generator in source_group['generators']:
                for split, count in source_group['sampling'].items():
                    split_dir_name = 'valid' if split == 'val' else split
                    source_dir = master_root / str(base_dir) / f"{generator}-{base_dir}" / split_dir_name

                    if not source_dir.is_dir():
                        continue

                    label_files = sorted(source_dir.glob('*_label.json'))
                    selected_labels = label_files[:count]

                    for label_file in selected_labels:
                        graph_id = label_file.name.replace('_label.json', '')
                        for suffix in ['_label.json', '_features.npy', '_edges.npz']:
                            src_file = source_dir / (graph_id + suffix)
                            if src_file.exists():
                                dest_file = local_paths[split] / src_file.name
                                files_to_copy.append((src_file, dest_file))

    if not files_to_copy:
        print(
            f"[!] Warning: no source files were found for '{mix_id}'. "
            f"Please check 'master_data_root' and the directory structure."
        )
    else:
        print(f"[*] Copying {len(files_to_copy)} files from master storage...")
        for src, dest in tqdm(files_to_copy, desc=f"Copying data ({mix_id})"):
            shutil.copy(src, dest)

    print(f"[+] Dynamic dataset '{mix_id}' was created successfully.")
    return {
        'train_path': str(local_paths['train']),
        'val_path': str(local_paths['val']),
        'test_path': str(local_paths['test']),
        '_temp_dir_to_cleanup': str(local_temp_root)
    }


def cleanup_dynamic_source_dirs(temp_dir_path):
    """Remove temporary directories created for dynamic-source datasets."""
    temp_dir = Path(temp_dir_path)
    if temp_dir.exists():
        print(f"[*] Cleaning dynamic dataset temporary directory: {temp_dir.resolve()}")
        shutil.rmtree(temp_dir)


# ==============================================================================
# Log parsing and Excel summarization
# ==============================================================================
def summarize(directory, campaign_name, all_param_keys):
    """
    Scan all logs, infer experiment groupings, and generate a compact Excel
    summary report containing core metrics only.
    """
    print(f"\n[*] Scanning and summarizing logs for campaign '{campaign_name}': {directory}")

    base_dir = Path(directory)
    log_files = [
        f for f in base_dir.rglob('*.log')
        if not f.name.endswith('_FAILED.log') and '.ipynb_checkpoints' not in f.parts
    ]

    if not log_files:
        print("[!] No valid log files (*.log) were found in the directory.")
        return

    all_results = []
    print("[*] Parsing all experiment logs...")

    for log_filepath in log_files:
        info = parse_log_info(log_filepath, base_dir, all_param_keys)
        metrics_list = parse_log_content(log_filepath)
        if info and metrics_list:
            for metrics in metrics_list:
                all_results.append({**info, **metrics})

    if not all_results:
        print("[!] All logs were parsed, but no valid results were extracted.")
        return

    df = pd.DataFrame(all_results)
    output_filename = base_dir / f"summary_report_{campaign_name}.xlsx"

    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            print("\n[*] Aggregating results by scenario and generating Excel sheets...")

            desired_metrics = ['Test MAE', 'Test RMSE', 'Test R2']

            for sheet_name, group_df in df.groupby('sheet_name'):
                print(f"  - Processing sheet: {sheet_name}")

                metric_cols = [col for col in desired_metrics if col in group_df.columns]
                if not metric_cols:
                    print(f"  - [Warning] No desired metrics were found in {sheet_name}. Skipping this sheet.")
                    continue

                param_cols = sorted([
                    p.split('.')[-1] for p in all_param_keys
                    if p.split('.')[-1] in group_df.columns
                ])

                has_varied_params = False
                if param_cols and group_df[param_cols].nunique().prod() > 1:
                    has_varied_params = True

                if has_varied_params:
                    summary_df = group_df.groupby(param_cols)[metric_cols].agg(['mean', 'std'])
                    summary_df.columns = [f"{col[0]} ({col[1]})" for col in summary_df.columns]
                    summary_df = summary_df.reset_index()
                else:
                    summary_stats = group_df[metric_cols].agg(['mean', 'std'])
                    if not summary_stats.empty:
                        s = summary_stats.unstack()
                        summary_df = s.to_frame().T
                        summary_df.columns = [f'{metric} ({stat})' for metric, stat in s.index]
                    else:
                        summary_df = pd.DataFrame()

                sort_col = 'Test MAE (mean)'
                if sort_col in summary_df.columns:
                    summary_df = summary_df.sort_values(by=sort_col, ascending=True)

                safe_sheet_name = re.sub(r'[\\/*?:\[\]]', '_', str(sheet_name))[:31]
                summary_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

        print(f"\n[+] Compact summary report generated successfully: {output_filename.resolve()}")
    except Exception as e:
        print(f"\n[!] Failed to save the Excel report: {e}")


def parse_log_content(filepath):
    """Extract test metrics from a training log file."""
    results_list = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            test_matches = re.findall(
                r"Test results - Loss: [\d.eE+-]+, RMSE: ([\d.eE+-]+), MAE: ([\d.eE+-]+), R²: ([\d.eE+-]+)",
                content
            )
            for match in test_matches:
                metrics = {
                    'Test RMSE': float(match[0]),
                    'Test MAE': float(match[1]),
                    'Test R2': float(match[2])
                }
                results_list.append(metrics)
        return results_list
    except Exception:
        return None


def parse_log_info(filepath, base_dir, all_param_keys):
    """Infer metadata and parameter settings from a log file path and filename."""
    info = {}
    try:
        relative_path_parts = filepath.relative_to(base_dir).parts
        is_param_search = bool(re.match(r'exp_\d+_.+', filepath.name))

        if is_param_search:
            info['sheet_name'] = relative_path_parts[0] if len(relative_path_parts) > 1 else "param_search_results"
            base_name = re.sub(r'^exp_\d+_', '', filepath.stem)
            key_map = {
                ''.join(filter(str.isalnum, key.replace('.', ''))).lower(): key
                for key in all_param_keys
            }

            parts = base_name.split('_')
            i = 0
            while i < len(parts) - 1:
                short_key, value_str = parts[i], parts[i + 1]
                full_key = key_map.get(short_key)
                if full_key:
                    param_name = full_key.split('.')[-1]
                    try:
                        if '.' in value_str or 'e-' in value_str.lower():
                            info[param_name] = float(value_str)
                        else:
                            info[param_name] = int(value_str)
                    except ValueError:
                        info[param_name] = str(value_str)
                    i += 2
                else:
                    i += 1
        else:
            if len(relative_path_parts) > 2:
                info['sheet_name'] = f"{relative_path_parts[0]}-{relative_path_parts[1]}"
            else:
                info['sheet_name'] = relative_path_parts[0] if relative_path_parts else "batch_run_results"

        return info
    except Exception:
        return None


# ==============================================================================
# Main execution workflow
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML experiment configuration file")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    print("[*] Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 to improve memory allocation behavior.")

    print("[*] Running pre-launch cleanup...")
    pre_cleanup_dirs = [Path('temp_data'), Path('cache')]
    for dir_to_clean in pre_cleanup_dirs:
        if dir_to_clean.exists():
            print(f"  - Removing existing directory: {dir_to_clean}")
            shutil.rmtree(dir_to_clean)
    print("[+] Pre-launch cleanup completed.")

    config_path = Path(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        master_config = yaml.safe_load(f)

    master_data_root = Path(master_config.get('master_data_root', '.'))
    print(f"[*] Master dataset root: {master_data_root.resolve()}")

    base_config = master_config.get('base_config', {})
    experiments = master_config.get('experiments', [])

    campaign_name = config_path.stem.replace('train-', '')
    print(f"====== Starting campaign: {campaign_name} ======")

    base_output_root = Path(base_config.get('output', {}).get('output_dir', 'outputs'))
    base_model_root = Path(base_config.get('output', {}).get('model_dir', 'models'))
    main_output_dir = base_output_root / f"{campaign_name}-outputs"
    main_model_dir = base_model_root / f"{campaign_name}-models"

    all_tasks = []
    all_param_keys = set()

    for exp_block in experiments:
        exp_params = copy.deepcopy(base_config)

        for key, value in exp_block.get('params', {}).items():
            section, param = key.split('.')
            exp_params.setdefault(section, {})[param] = value

        param_combinations = []
        param_grid = exp_block.get('param_grid', {})
        ablation_grid = exp_block.get('ablation_grid', {})
        sensitivity_grid = exp_block.get('sensitivity_grid', {})

        if param_grid:
            print("[*] Detected 'param_grid'. Running grid search...")
            all_param_keys.update(param_grid.keys())
            keys, values = zip(*param_grid.items())
            param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        elif ablation_grid or sensitivity_grid:
            grid_type = "ablation_grid" if ablation_grid else "sensitivity_grid"
            active_grid = ablation_grid or sensitivity_grid
            print(f"[*] Detected '{grid_type}'. Running single-variable analysis...")

            all_param_keys.update(active_grid.keys())
            for key, value_list in active_grid.items():
                for value in value_list:
                    param_combinations.append({key: value})

        if not param_combinations and (param_grid or ablation_grid or sensitivity_grid):
            pass
        elif not param_combinations:
            param_combinations = [{}]

        data_contexts = []
        dataset_def = exp_block['dataset']
        data_type = dataset_def.get('type')

        if data_type in ['batch', 'multi_source', 'dynamic_source']:
            templates = dataset_def.get('templates', {})

            for instance in dataset_def.get('instances', []):
                format_vars = {k: v for k, v in templates.items() if isinstance(v, str)}
                format_vars.update(instance)

                if data_type == 'dynamic_source':
                    dataset_id = instance.get('mix_id')
                    context = {'dataset_id': dataset_id, '_is_dynamic_source': True}
                    dynamic_paths = prepare_dynamic_source_dirs(instance, master_data_root, templates)
                    context['_temp_dir_to_cleanup'] = dynamic_paths.pop('_temp_dir_to_cleanup')

                    for key, path in dynamic_paths.items():
                        section, param = ('dataset', key)
                        context.setdefault(section, {})[param] = path
                else:
                    dataset_id_key = dataset_def.get('id_key')
                    dataset_id = instance.get(dataset_id_key)
                    if not dataset_id and dataset_id_key in templates:
                        dataset_id = templates[dataset_id_key].format(**format_vars)
                    context = {'dataset_id': dataset_id}

                if 'subdir_key' in dataset_def:
                    context['_subdir'] = instance.get(dataset_def['subdir_key'])

                if data_type == 'multi_source':
                    context['_is_multi_source'] = True
                    multi_source_config_instance = {}

                    for key in ['train_source_dirs', 'val_source_dirs', 'test_source_dirs']:
                        if key in templates:
                            multi_source_config_instance[key] = [
                                path_template.format(**format_vars)
                                for path_template in templates[key]
                            ]

                    if 'temp_data_dir' in templates:
                        multi_source_config_instance['temp_data_dir'] = templates['temp_data_dir'].format(**format_vars)
                    else:
                        temp_dir_name = f"temp_data_{dataset_id}_{context.get('_subdir', '')}".replace(' ', '_')
                        multi_source_config_instance['temp_data_dir'] = temp_dir_name

                    context['_multi_source_config'] = multi_source_config_instance

                for key_path, template in templates.items():
                    special_keys = ['train_source_dirs', 'val_source_dirs', 'test_source_dirs', 'temp_data_dir']
                    if data_type != 'dynamic_source':
                        special_keys.append(dataset_def.get('id_key'))

                    if key_path in special_keys:
                        continue

                    if isinstance(template, str) and '.' in key_path:
                        section, param = key_path.split('.', 1)
                        context.setdefault(section, {})[param] = template.format(**format_vars)

                data_contexts.append(context)

        for data_ctx in data_contexts:
            for param_combo in param_combinations:
                task = copy.deepcopy(exp_params)

                for section, params in data_ctx.items():
                    if section.startswith('_'):
                        continue
                    if isinstance(params, dict):
                        task.setdefault(section, {}).update(params)

                for key, value in param_combo.items():
                    section, param = key.split('.')
                    task.setdefault(section, {})[param] = value

                task['_metadata'] = {'param_combo': param_combo, **data_ctx}
                all_tasks.append(task)

    print(f"\n[+] Task generation completed: {len(all_tasks)} independent experiment tasks.")
    print(f"[+] Models will be saved to: {main_model_dir.resolve()}")
    print(f"[+] Logs and results will be saved to: {main_output_dir.resolve()}")

    print("\n" + "=" * 80)
    print("--- Starting all tasks ---")
    print("=" * 80)

    completed_temp_dirs = set()

    for i, task in enumerate(all_tasks):
        metadata = task.pop('_metadata')
        param_combo = metadata['param_combo']

        if metadata.get('_is_multi_source', False):
            paths = setup_symlinked_dirs(metadata['_multi_source_config'])
            task['dataset'].update(paths)

        exp_name_parts = [f"exp_{i + 1:03d}"]
        for key, value in sorted(param_combo.items()):
            short_key = ''.join(filter(str.isalnum, key.replace('.', ''))).lower()
            exp_name_parts.append(f"{short_key}_{value}")
        sub_experiment_name = "_".join(exp_name_parts)

        dataset_path_part = Path(str(metadata['dataset_id']))
        if '_subdir' in metadata and not param_combo:
            dataset_path_part = dataset_path_part / str(metadata['_subdir'])

        task['output']['output_dir'] = str(main_output_dir / dataset_path_part)
        task['output']['model_dir'] = str(main_model_dir / dataset_path_part)

        command = [PYTHON_EXECUTABLE, TRAIN_SCRIPT, '--experiment_name', sub_experiment_name]

        flat_task = {}
        for section, params in task.items():
            if isinstance(params, dict):
                flat_task.update(params)

        for key, value in flat_task.items():
            if value is not None:
                command.extend([f'--{key}', str(value)])

        log_dir = Path(task['output']['output_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_filename = log_dir / f"{sub_experiment_name}.log"

        print(f"\n--- [{i + 1}/{len(all_tasks)}] Starting: {sub_experiment_name} ---")
        print(f"      Dataset ID: {metadata['dataset_id']}")
        print(f"      Log file: {log_filename}")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1
            )

            with open(log_filename, 'w', encoding='utf-8') as log_file:
                for line in process.stdout:
                    print(line, end='')
                    log_file.write(line)

            return_code = process.wait()
            if return_code != 0:
                print(f"\n[!] Task {sub_experiment_name} failed with return code: {return_code}")
                os.rename(log_filename, str(log_filename) + '_FAILED.log')
            else:
                print(f"[+] Task {sub_experiment_name} completed successfully.")
        except Exception as e:
            print(f"\n[!] A fatal error occurred while starting or running task {sub_experiment_name}: {e}")

        if metadata.get('_is_multi_source', False):
            completed_temp_dirs.add(metadata['_multi_source_config']['temp_data_dir'])
        if metadata.get('_is_dynamic_source', False):
            completed_temp_dirs.add(metadata['_temp_dir_to_cleanup'])

    print("\n" + "=" * 80)
    print(f"[+] Campaign '{campaign_name}' finished.")

    print("\n[*] Running post-execution cleanup...")
    if completed_temp_dirs:
        print("  - Cleaning temporary data directories...")
        for temp_dir_str in completed_temp_dirs:
            cleanup_dynamic_source_dirs(temp_dir_str)

    cache_dir = Path('cache')
    if cache_dir.exists():
        print(f"  - Cleaning cache directory: {cache_dir.resolve()}")
        shutil.rmtree(cache_dir)

    print("[+] Post-execution cleanup completed.")
    print("=" * 80)

    summarize(main_output_dir, campaign_name, all_param_keys)


if __name__ == "__main__":
    main()
