#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Manager Module

Handles loading and managing segmented TCR-GIN models.
"""

import re
import zipfile
import yaml
import json
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

from webapp.core.tcr_gin import TCR_GIN
from argparse import Namespace
from torch_geometric.data import Batch


DEFAULT_SCALE_FACTOR = 1.0
MODEL_INDEX_TO_USE = 0

PARAM_KEY_MAP = {
    'modelactivationfn': 'activation_fn',
    'modeljktype': 'jk_type',
    'modelusevirtualnode': 'use_virtual_node',
    'modeluseresidual': 'use_residual',
    'pissconsistencylambda': 'consistency_lambda',
    'pisspissk': 'piss_k',
    'modelfeaturedim': 'feature_dim',
}

REQUIRED_MODEL_PARAMS = [
    'input_dim', 'hidden_dim', 'num_layers', 'dropout',
    'jk_type', 'use_virtual_node', 'use_residual', 'activation_fn'
]


def _convert_config_value(value: Any) -> Any:
    """Convert YAML/string config values to model constructor friendly types."""
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower == 'true':
            return True
        if lower == 'false':
            return False
        try:
            return float(value) if '.' in value or 'e' in lower else int(value)
        except ValueError:
            return value
    return value


def _normalise_model_config(config_yaml: dict) -> dict:
    """Extract TCR-GIN model parameters from supported training/evaluation configs."""
    if not isinstance(config_yaml, dict):
        raise ValueError("The configuration file is empty or invalid.")

    if 'base_model_params' in config_yaml:
        model_config = dict(config_yaml.get('base_model_params') or {})
    elif 'base_config' in config_yaml:
        base_config = config_yaml.get('base_config') or {}
        model_config = dict(base_config.get('model') or {})
        training_config = base_config.get('training') or {}
        if 'label_scale_factor' in training_config:
            model_config['label_scale_factor'] = training_config['label_scale_factor']
    else:
        model_config = dict(config_yaml)
        if 'model' in config_yaml and isinstance(config_yaml['model'], dict):
            model_config.update(config_yaml['model'])
        if 'training' in config_yaml and isinstance(config_yaml['training'], dict):
            training_config = config_yaml['training']
            if 'label_scale_factor' in training_config:
                model_config['label_scale_factor'] = training_config['label_scale_factor']

    final_config = {}
    for key, value in model_config.items():
        mapped_key = PARAM_KEY_MAP.get(str(key).lower(), key)
        final_config[mapped_key] = _convert_config_value(value)

    # The early-warning evaluation scripts use feature_dim as the effective input_dim.
    if final_config.get('feature_dim') is not None:
        final_config['input_dim'] = int(final_config['feature_dim'])

    final_config['label_scale_factor'] = float(
        final_config.get('label_scale_factor', DEFAULT_SCALE_FACTOR)
    )
    return final_config


def _configured_ranges(config_yaml: dict) -> list[Tuple[int, int]]:
    """Read model_suite node ranges from an early-warning evaluation config."""
    ranges = []
    if not isinstance(config_yaml, dict):
        return ranges
    for item in config_yaml.get('model_suite', []) or []:
        node_range = item.get('node_range') if isinstance(item, dict) else None
        if isinstance(node_range, (list, tuple)) and len(node_range) == 2:
            try:
                ranges.append((int(node_range[0]), int(node_range[1])))
            except Exception:
                continue
    return ranges


def _configured_model_entries(config_yaml: dict) -> list[dict]:
    """Read model_suite entries with node ranges and base_dir names."""
    entries = []
    if not isinstance(config_yaml, dict):
        return entries
    for item in config_yaml.get('model_suite', []) or []:
        if not isinstance(item, dict):
            continue
        node_range = item.get('node_range')
        if not isinstance(node_range, (list, tuple)) or len(node_range) != 2:
            continue
        try:
            range_tuple = (int(node_range[0]), int(node_range[1]))
        except Exception:
            continue
        base_dir = item.get('base_dir')
        entries.append({
            'range': range_tuple,
            'base_name': Path(str(base_dir)).name if base_dir else None,
            'name': item.get('name', ''),
        })
    return entries


def _training_mix_entries(config_yaml: dict) -> list[dict]:
    """Read model-family names from training configs that contain dataset mix_id values."""
    if not isinstance(config_yaml, dict):
        return []

    entries = []
    seen = set()
    for experiment in config_yaml.get('experiments', []) or []:
        if not isinstance(experiment, dict):
            continue
        dataset = experiment.get('dataset') or {}
        if not isinstance(dataset, dict):
            continue
        for instance in dataset.get('instances', []) or []:
            if not isinstance(instance, dict):
                continue
            mix_id = instance.get('mix_id')
            if not mix_id or mix_id in seen:
                continue
            inferred_range = _range_from_name(str(mix_id))
            if inferred_range is None:
                continue
            entries.append({
                'range': inferred_range,
                'base_name': str(mix_id),
                'name': str(mix_id),
                'strict_range': _is_strict_range_name(str(mix_id)),
            })
            seen.add(mix_id)

    entries.sort(key=lambda item: item['range'][0])
    if entries and entries[0]['range'][0] > 0 and not entries[0].get('strict_range', False):
        _, end = entries[0]['range']
        entries[0]['range'] = (0, end)

    for idx in range(len(entries) - 1):
        start, _ = entries[idx]['range']
        next_start, _ = entries[idx + 1]['range']
        entries[idx]['range'] = (start, next_start)
    return entries


def _range_from_name(name: str) -> Optional[Tuple[int, int]]:
    """Infer a model range from a segment directory name."""
    lower = name.lower()
    if (
        lower.startswith('exp_')
        or lower.startswith('model_run')
        or lower in {'model', 'models', 'checkpoint', 'checkpoints'}
        or 'multisource' in lower
    ):
        return None
    numbers = [int(x) for x in re.findall(r'\d+', name)]
    if not numbers:
        return None
    if re.fullmatch(r'\d+\s*-\s*\d+', name):
        return numbers[0], numbers[1]
    if len(numbers) == 1:
        # Training directories like "300-transport" represent the 300-scale segment.
        return numbers[0], numbers[0] + 100
    return min(numbers), max(numbers) + 100


def _range_from_path(path: Path) -> Optional[Tuple[int, int]]:
    """Infer a segment range from any directory component such as 100-200."""
    for part in reversed(path.parts):
        inferred = _range_from_name(part)
        if inferred is not None:
            return inferred
    return None


def _is_strict_range_name(name: str) -> bool:
    """Return True for old-style directory names such as 0-100 or 100-200."""
    return bool(re.fullmatch(r'\d+\s*-\s*\d+', name))


def _segment_family_from_checkpoint(model_root: Path, checkpoint: Path) -> Optional[str]:
    """Find the model-family directory that owns a checkpoint."""
    try:
        rel_parts = checkpoint.relative_to(model_root).parts
    except ValueError:
        rel_parts = checkpoint.parts

    if len(rel_parts) < 2:
        return None

    # Experiment outputs usually look like:
    # <family>/exp_001/model_run_1.pt. Pick the directory before exp_*.
    for idx, part in enumerate(rel_parts[:-1]):
        if part.lower().startswith('exp_') and idx > 0:
            return rel_parts[idx - 1]

    # Otherwise prefer the deepest directory whose name looks like a segment.
    for part in reversed(rel_parts[:-1]):
        if _range_from_name(part) is not None:
            return part

    return rel_parts[-2]


def _matches_base_name(path: Path, base_name: Optional[str]) -> bool:
    """Return True when a checkpoint lives under the configured base_dir name."""
    if not base_name:
        return False
    return any(part == base_name for part in path.parts)


def _choose_model_file(model_files: list[Path]) -> Optional[Path]:
    """Match the early-warning scripts' MODEL_INDEX_TO_USE selection rule."""
    if not model_files:
        return None
    model_files = sorted(model_files)
    if len(model_files) <= MODEL_INDEX_TO_USE:
        return model_files[-1]
    return model_files[MODEL_INDEX_TO_USE]


def _infer_entries_from_model_root(model_root: Path) -> list[dict]:
    """Infer one entry per model-family directory when config has no model_suite."""
    entries = []
    seen = set()
    for checkpoint in sorted(model_root.rglob('*.pt')):
        family = _segment_family_from_checkpoint(model_root, checkpoint)
        if family in seen:
            continue
        inferred_range = _range_from_name(family)
        if inferred_range is None:
            continue
        entries.append({
            'range': inferred_range,
            'base_name': family,
            'name': family,
            'strict_range': _is_strict_range_name(family),
        })
        seen.add(family)

    entries.sort(key=lambda item: item['range'][0])

    if entries and entries[0]['range'][0] > 0 and not entries[0].get('strict_range', False):
        _, end = entries[0]['range']
        entries[0]['range'] = (0, end)

    # Extend preceding segments to the next segment start so there are no gaps.
    for idx in range(len(entries) - 1):
        start, _ = entries[idx]['range']
        next_start, _ = entries[idx + 1]['range']
        entries[idx]['range'] = (start, next_start)

    return entries


def _normalise_overlapping_ranges(entries: list[dict]) -> list[dict]:
    """Make ranges non-overlapping while preserving configured order."""
    if not entries:
        return entries
    sorted_entries = sorted(entries, key=lambda item: item['range'][0])
    for idx in range(len(sorted_entries) - 1):
        start, end = sorted_entries[idx]['range']
        next_start = sorted_entries[idx + 1]['range'][0]
        if end > next_start:
            sorted_entries[idx]['range'] = (start, next_start)
    return sorted_entries


def _read_scale_factor(checkpoint_path: Path, default: float) -> float:
    """Find label_stats.json near a checkpoint and return its scale factor."""
    candidates = [
        checkpoint_path.parent / 'label_stats.json',
        checkpoint_path.parent.parent / 'label_stats.json',
        checkpoint_path.parent.parent / 'outputs' / 'label_stats.json',
    ]

    for path in candidates:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return float(json.load(f).get('scale_factor', default))
            except Exception:
                continue
    return float(default)


class SegmentedModel:
    """Wrapper for segmented TCR-GIN models."""

    def __init__(self, segments: Dict[Tuple[int, int], Dict[str, Any]], config: dict):
        """
        Initialize segmented model.

        Args:
            segments: Dictionary mapping (start, end) ranges to model metadata
            config: Model configuration dictionary
        """
        self.segments = segments
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move all models to device
        for segment in self.segments.values():
            segment['model'].to(self.device)
            segment['model'].eval()

    @property
    def input_dim(self) -> int:
        """Effective node-feature dimension expected by the loaded models."""
        return int(self.config.get('input_dim', self.config.get('feature_dim', 7)))

    def get_input_dim(self) -> int:
        return self.input_dim

    def _select_segment(self, current_nodes: int) -> Dict[str, Any]:
        """Select the model segment using the same node-range rule as evaluation."""
        for (start, end), segment in sorted(self.segments.items()):
            if end == -1:
                if current_nodes >= start:
                    return segment
            elif start <= current_nodes < end:
                return segment

        closest_range = min(
            self.segments.keys(),
            key=lambda r: min(abs(r[0] - current_nodes), abs(r[1] - current_nodes))
        )
        return self.segments[closest_range]

    def predict(self, graph_data, current_nodes: int) -> float:
        """
        Predict using appropriate segment based on current network size.

        Args:
            graph_data: PyTorch Geometric Data object
            current_nodes: Current number of nodes in network

        Returns:
            Predicted critical threshold
        """
        if graph_data is None or current_nodes <= 0:
            return 0.0

        segment = self._select_segment(current_nodes)
        model = segment['model']
        scale_factor = float(segment.get('scale_factor', DEFAULT_SCALE_FACTOR))

        batch = Batch.from_data_list([graph_data]).to(self.device)
        with torch.no_grad():
            pred_scaled = model(batch) / scale_factor
            prediction = torch.clamp(pred_scaled, 0.0, 1.0).view(-1)[0].item()

        return float(prediction)

    def predict_many(self, graph_items: list[tuple[Any, int]]) -> list[float]:
        """Predict critical thresholds for multiple component graphs."""
        if not graph_items:
            return []

        results: list[Optional[float]] = [None] * len(graph_items)
        grouped: dict[int, list[tuple[int, Any]]] = {}
        segment_lookup = {}

        for idx, (graph_data, current_nodes) in enumerate(graph_items):
            if graph_data is None or current_nodes <= 0:
                results[idx] = 0.0
                continue
            segment = self._select_segment(current_nodes)
            segment_key = id(segment)
            grouped.setdefault(segment_key, []).append((idx, graph_data))
            segment_lookup[segment_key] = segment

        with torch.no_grad():
            for segment_key, indexed_graphs in grouped.items():
                segment = segment_lookup[segment_key]
                model = segment['model']
                scale_factor = float(segment.get('scale_factor', DEFAULT_SCALE_FACTOR))
                batch = Batch.from_data_list([item[1] for item in indexed_graphs]).to(self.device)
                pred_scaled = torch.clamp(model(batch) / scale_factor, 0.0, 1.0).view(-1).detach().cpu().tolist()
                for (result_idx, _), pred_value in zip(indexed_graphs, pred_scaled):
                    results[result_idx] = float(pred_value)

        return [0.0 if value is None else float(value) for value in results]

    def predict_collapse_distance(self, graph_data, current_nodes: int, initial_nodes: int) -> float:
        """
        Return Collapse Distance exactly as used in calculate_decision_window.py:
        critical_threshold * (current_nodes / initial_nodes).
        """
        if initial_nodes <= 0:
            return 0.0
        critical_threshold = self.predict(graph_data, current_nodes)
        return float(critical_threshold * (current_nodes / initial_nodes))

    def get_info(self) -> Dict:
        """Get model information."""
        visible_config = {
            key: value for key, value in self.config.items()
            if key != 'label_scale_factor'
        }
        return {
            'num_segments': len(self.segments),
            'segments': [f"{start}-{end}" for start, end in sorted(self.segments.keys())],
            'checkpoints': {
                f"{start}-{end}": "Loaded"
                for (start, end), segment in sorted(self.segments.items())
            },
            'device': str(self.device),
            'config': visible_config
        }


class ModelManager:
    """Manage model loading and validation."""

    @staticmethod
    def load_from_directory(model_root: str, config_path: str) -> SegmentedModel:
        """
        Load segmented model checkpoints from an existing directory.

        This is useful for local runs where the original experiments directory is
        available and avoids requiring a zip file with specific folder names.
        """
        model_root_path = Path(model_root)
        if not model_root_path.exists() or not model_root_path.is_dir():
            raise ValueError(f"Model directory does not exist: {model_root}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_yaml = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Could not read the configuration file: {e}")

        return ModelManager._load_from_root(model_root_path, config_yaml)

    @staticmethod
    def load_from_zip(zip_path: str, config_path: str, temp_dir: str = 'temp_models') -> SegmentedModel:
        """
        Load segmented model from ZIP file.

        Args:
            zip_path: Path to model ZIP file
            config_path: Path to YAML config file
            temp_dir: Temporary directory for extraction

        Returns:
            SegmentedModel object
        """
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        marker_path = temp_path / '.extract_signature'
        zip_source = Path(zip_path)
        try:
            zip_stat = zip_source.stat()
            extract_signature = f"{zip_source.resolve()}|{zip_stat.st_size}|{zip_stat.st_mtime_ns}"
        except OSError:
            extract_signature = str(zip_source)

        # Extract ZIP only when the source changed.
        if not marker_path.exists() or marker_path.read_text(encoding='utf-8', errors='ignore') != extract_signature:
            try:
                import shutil
                for child in temp_path.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
                marker_path.write_text(extract_signature, encoding='utf-8')
            except Exception as e:
                raise ValueError(f"Could not extract the model archive: {e}")

        # Load config
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_yaml = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Could not read the configuration file: {e}")

        return ModelManager._load_from_root(temp_path, config_yaml)

    @staticmethod
    def _load_from_root(model_root: Path, config_yaml: dict) -> SegmentedModel:
        """Shared implementation for zip extraction directories and local model roots."""
        model_config = _normalise_model_config(config_yaml)
        configured_ranges = _configured_ranges(config_yaml)
        configured_entries = _normalise_overlapping_ranges(_configured_model_entries(config_yaml))
        entries_from_config = bool(configured_entries)

        # Ensure required parameters exist
        missing = [p for p in REQUIRED_MODEL_PARAMS if p not in model_config]
        if missing:
            raise ValueError(f"Configuration is missing required parameters: {', '.join(missing)}")

        # Scan for model segments
        segments = {}

        model_files = sorted(model_root.rglob('*.pt'))
        selected_models = []

        if not configured_entries:
            configured_entries = _training_mix_entries(config_yaml)

        if not configured_entries:
            configured_entries = _infer_entries_from_model_root(model_root)

        used_paths = set()
        for entry in configured_entries:
            candidates = [
                path for path in model_files
                if _matches_base_name(path, entry['base_name']) and path not in used_paths
            ]
            target = _choose_model_file(candidates)
            if target is None:
                continue
            selected_models.append((target, entry['range']))
            used_paths.add(target)

        if entries_from_config and len(selected_models) < len(configured_entries):
            missing = [
                entry.get('base_name') or entry.get('name') or str(entry.get('range'))
                for entry in configured_entries
                if not any(
                    selected_range == entry['range']
                    for _, selected_range in selected_models
                )
            ]
            inferred_entries = _infer_entries_from_model_root(model_root)
            unused_inferred = [
                entry for entry in inferred_entries
                if not any(_matches_base_name(path, entry['base_name']) for path, _ in selected_models)
            ]
            missing_entries = [
                entry for entry in configured_entries
                if not any(
                    selected_range == entry['range']
                    for _, selected_range in selected_models
                )
            ]
            for entry, inferred in zip(missing_entries, unused_inferred):
                candidates = [
                    path for path in model_files
                    if _matches_base_name(path, inferred['base_name']) and path not in used_paths
                ]
                target = _choose_model_file(candidates)
                if target is not None:
                    selected_models.append((target, entry['range']))
                    used_paths.add(target)

            if len(selected_models) < len(configured_entries):
                raise ValueError(
                    "Not all model segments required by the configuration were found.\n\n"
                    f"Configured segments: {len(configured_entries)}\n"
                    f"Matched segments: {len(selected_models)}\n"
                    f"Unmatched entries: {', '.join(missing)}\n\n"
                    "Check that the model directory or ZIP contains checkpoints matching "
                    "model_suite[*].base_dir."
                )

        if not selected_models:
            seen_ranges = set()
            for idx, item in enumerate(model_files):
                node_range = _range_from_path(item)
                if node_range is None and idx < len(configured_ranges):
                    node_range = configured_ranges[idx]
                if node_range in seen_ranges:
                    continue
                selected_models.append((item, node_range))
                seen_ranges.add(node_range)

        for item, configured_range in selected_models:
            node_range = _range_from_path(item)
            if configured_range is not None:
                node_range = configured_range
            if node_range is None:
                continue

            start, end = node_range
            args = Namespace(**model_config)
            model = TCR_GIN(args)

            try:
                checkpoint = torch.load(item, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            except Exception as e:
                raise ValueError(f"Could not load model checkpoint {item.name}: {e}")

            model.eval()
            scale_factor = _read_scale_factor(
                item,
                default=float(model_config.get('label_scale_factor', DEFAULT_SCALE_FACTOR))
            )
            segments[(start, end)] = {
                'model': model,
                'path': str(item),
                'scale_factor': scale_factor,
            }

        if not segments:
            raise ValueError(
                "No valid model files were found.\n\n"
                "Make sure the archive or directory contains `.pt` checkpoints and that "
                "the final folder names referenced by model_suite[*].base_dir appear in "
                "the model paths. Without model_suite, the loader attempts to infer node "
                "ranges from folder names such as 100-200-transport."
            )

        return SegmentedModel(segments, model_config)

    @staticmethod
    def validate_config_match(model: SegmentedModel, config_path: str) -> Tuple[bool, str]:
        """
        Validate if model and config are compatible.

        Args:
            model: SegmentedModel object
            config_path: Path to config file

        Returns:
            - True if compatible
            - Description message
        """
        # Basic validation: check if model loaded successfully
        if len(model.segments) == 0:
            return False, "Model loading failed: no valid model segment was found."

        # Check config parameters
        config = model.config
        required = ['input_dim', 'hidden_dim', 'num_layers']

        for param in required:
            if param not in config:
                return False, f"Configuration is missing parameter: {param}"

        return True, f"Configuration matched. Loaded {len(model.segments)} model segment(s)."
