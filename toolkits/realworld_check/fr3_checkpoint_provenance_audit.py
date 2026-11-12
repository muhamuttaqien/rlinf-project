# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Inspect local sim checkpoint provenance without loading weights or using ROS.

An action vector's width does not establish its units, coordinate frame or
gripper meaning. This audit reads only bounded textual metadata and norm
statistics; it does not compute commands or claim FR3 normalization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

SCHEMA = "fr3_checkpoint_provenance_review_v1"
CANDIDATE_FR3_STATE_DIM = 19
CANDIDATE_FR3_ACTION_DIM = 7
RELEVANT_CONFIG_NAMES = {
    "config.yaml", "config.yml", "config.json",
    "train_config.yaml", "train_config.yml", "train_config.json",
    "training_config.yaml", "training_config.json",
    "args.json", "args.yaml", "params.json", "parameters.yaml",
    "hydra.yaml", "overrides.yaml", "info.json",
}
MAX_CONFIG_FILES = 20
MAX_SCANNED_DIRS = 700
MAX_DEPTH = 8
MAX_TEXT_BYTES = 256_000
MATCHES_PER_FILE = 12
FIELD_PATTERN = re.compile(
    r"(?i)\b("
    r"config_name|repo_id|dataset_path|train_data_paths|norm_stats_path|"
    r"control_mode|action_dim|action_horizon|action_env_dim|"
    r"state_dim|num_action_chunks|task_description|"
    r"extra_delta_transform|action_space|policy_setup|robot_model"
    r")\b"
)


def norm_dimensions(path: Path) -> dict[str, Any]:
    """Validate structural consistency, but NEVER apply simulator statistics."""
    source = Path(path).expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    data = raw.get("norm_stats", raw)
    if not isinstance(data, dict):
        raise ValueError("Normalization JSON must be an object")
    summary: dict[str, Any] = {}
    for name in ("state", "actions", "action"):
        item = data.get(name)
        if item is None:
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{name}: expected statistics mapping")
        dimensions = {}
        for field in ("mean", "std", "q01", "q99", "min", "max"):
            if field not in item:
                continue
            values = item[field]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{name}.{field}: expected nonempty numeric list")
            try:
                clean = [float(value) for value in values]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name}.{field}: expected numeric values") from exc
            if not all(math.isfinite(number) for number in clean):
                raise ValueError(f"{name}.{field}: nonfinite value")
            dimensions[field] = len(clean)
        if not dimensions or len(set(dimensions.values())) != 1:
            raise ValueError(f"{name}: missing or inconsistent statistics dimensions")
        summary[name] = {
            "dim": next(iter(dimensions.values())),
            "fields": sorted(dimensions),
        }
    state_dim = summary.get("state", {}).get("dim")
    action_dim = summary.get("actions", summary.get("action", {})).get("dim")
    return {
        "file": str(source),
        "statistics": summary,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_matches_candidate_FR3": state_dim == CANDIDATE_FR3_STATE_DIM,
        "action_width_matches_candidate_FR3": action_dim == CANDIDATE_FR3_ACTION_DIM,
        "sim_stats_applied": False,
        "approved_for_FR3": False,
        "action_semantics_verified": False,
    }


def scan_experiment_metadata(root: Path) -> dict[str, Any]:
    """Bounded search of metadata stored in the ACTUAL experiment output.

    Only textual config contents inside this root are inspected. Paths and
    matching lines are evidence for human follow-up, not verified semantics.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return {
            "root": str(root), "exists": False, "files": [],
            "scan_truncated": False, "directories_scanned": 0,
        }
    configs: list[dict[str, Any]] = []
    dirs_seen = 0
    truncated = False
    for parent, dirs, files in os.walk(root):
        dirs_seen += 1
        depth = len(Path(parent).relative_to(root).parts)
        if dirs_seen > MAX_SCANNED_DIRS:
            truncated = True
            break
        dirs[:] = sorted(
            name for name in dirs
            if name not in {"model_state_dict", "images", "videos", "frames",
                            "wandb", ".git", "__pycache__", "tfevents"}
        ) if depth < MAX_DEPTH else []
        for name in sorted(files):
            if name not in RELEVANT_CONFIG_NAMES and not (
                name.startswith("config") and name.endswith((".yml", ".yaml", ".json"))
            ):
                continue
            p = Path(parent) / name
            try:
                if p.stat().st_size > MAX_TEXT_BYTES:
                    continue
                content = p.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            matches = [
                line.strip()[:200] for line in content.splitlines()
                if FIELD_PATTERN.search(line)
            ][:MATCHES_PER_FILE]
            configs.append({
                "relative_path": str(p.relative_to(root)),
                "matched_lines": matches,
            })
            if len(configs) >= MAX_CONFIG_FILES:
                truncated = True
                break
        if truncated:
            break
    return {
        "root": str(root), "exists": True,
        "directories_scanned": min(dirs_seen, MAX_SCANNED_DIRS),
        "files": configs,
        "scan_truncated": truncated,
    }


def build_report(
    checkpoint: Path, norm_stats: Path, experiment_root: Path,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    if checkpoint.name != "full_weights.pt":
        raise ValueError("Expected RLinf actor full_weights.pt path")
    evidence = scan_experiment_metadata(experiment_root)
    normalization = norm_dimensions(norm_stats)
    return {
        "audit_schema": SCHEMA,
        "mode": "offline_read_only",
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_weights_loaded": False,
        "normalization": normalization,
        "experiment_metadata": evidence,
        "source_code_findings": {
            "isaaclab_stack_cube_output": "IsaacLabOutputs converts last channel with np.sign to {-1, 0, +1} (binary intended)",
            "realworld_output": "RealworldOutputs copies first seven channels without physical calibration",
            "maniskill_output": "ManiSkillOutputs copies first seven channels",
            "checkpoint_training_pipeline": "NOT ESTABLISHED from filename or array shape",
        },
        "requires_manual_provenance_check": [
            "Confirm experiment config and actual SFT dataset identity",
            "Confirm training state feature names/order, expected 18D normalization and source frames",
            "Confirm seven training action channels, units, frames, gripper convention and transform",
            "Confirm whether any config uses extra delta action transform",
            "Verify FR3-specific action labels and task-appropriate stats from new approved demonstrations",
        ],
        "model_action_semantics_verified": False,
        "FR3_normalization_ready": False,
        "robot_commands_sent": False,
        "physical_execution_approved": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        build_report(args.checkpoint, args.norm_stats, args.experiment_root),
        indent=2, allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
