#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal runner for collapse-profile YAML configs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml



def discover_configs(single_config: str | None, config_dir: str | None):
    if single_config:
        return [Path(single_config)]
    if not config_dir:
        raise ValueError("Provide either --config or --config_dir")
    return sorted(Path(config_dir).glob("*.yaml"))



def main():
    parser = argparse.ArgumentParser(description="Run collapse-profile experiments from YAML configs")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_dir", type=str, default=None)
    parser.add_argument("--python", type=str, default="python")
    args = parser.parse_args()

    config_paths = discover_configs(args.config, args.config_dir)
    if not config_paths:
        raise RuntimeError("No YAML config files found.")

    this_dir = Path(__file__).resolve().parent
    train_script = this_dir / "train_profile.py"

    for idx, config_path in enumerate(config_paths, start=1):
        print("=" * 80)
        print(f"[{idx}/{len(config_paths)}] Running {config_path.name}")
        print("=" * 80)
        cmd = [args.python, str(train_script), "--config", str(config_path)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
