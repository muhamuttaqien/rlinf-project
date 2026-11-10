# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Train-split-only OpenPI normalization for FR3 SpaceMouse REQUEST CANDIDATES.

Reads 19D state and 7D normalized operator requests from LeRobot Parquet,
without decoding camera images. Never reads validation data, runs training,
accesses ROS, or issues robot commands. This is an OFFLINE research artifact,
NOT a verified controller-action normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.realworld_check.fr3_request_lerobot_export import EXPORT_SCHEMA

MODEL_DIM = 32
STATE_DIM = 19
ACTION_DIM = 7
SCHEMA = "fr3_spacemouse_request_openpi_norm_candidate_v1"


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return result


def _stats(values: np.ndarray, width: int) -> dict[str, list[float]]:
    """Numerical quantiles over *unpadded* features, then identity pad to 32D."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width or len(values) < 2:
        raise ValueError(f"Expected at least two {width}D samples")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite state/action value")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
    # Keep constant-valued features numerically stable. These are
    # normalization-only safeguards, not synthesized training labels.
    constant = q99 - q01 < 1e-6
    q01[constant] = mean[constant] - 0.5
    q99[constant] = mean[constant] + 0.5
    std[std < 1e-6] = 1.0
    def pad(x, pad_value):
        return np.pad(x, (0, MODEL_DIM - width), constant_values=pad_value).tolist()
    return {
        "mean": pad(mean, 0.0),
        "std": pad(std, 1.0),
        "q01": pad(q01, 0.0),
        "q99": pad(q99, 1.0),
    }


def compute_from_arrays(state: np.ndarray, actions: np.ndarray) -> dict:
    """Isolated, deterministic stats calculation for unit tests and export."""
    state = np.asarray(state, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if (state.ndim != 2 or actions.ndim != 2
            or state.shape != (actions.shape[0], STATE_DIM)
            or actions.shape[1] != ACTION_DIM):
        raise ValueError("Expected aligned N x 19 states and N x 7 actions")
    if (not np.isfinite(actions).all()
            or np.any(np.abs(actions[:, :6]) > 1.001)
            or np.any(~np.isin(actions[:, 6], [0.0, 1.0]))):
        raise ValueError("Not a valid normalized SpaceMouse + binary grip dataset")
    return {
        "state": _stats(state, STATE_DIM),
        "actions": _stats(actions, ACTION_DIM),
    }


def compute_from_lerobot(dataset: Path) -> tuple[dict, dict]:
    dataset = dataset.expanduser().resolve()
    provenance_path = dataset / "fr3_command_request_provenance.json"
    provenance = _json(provenance_path)
    info = _json(dataset / "meta" / "info.json")
    if provenance.get("schema") != EXPORT_SCHEMA or provenance.get("source_split") != "train":
        raise ValueError("Training-only SpaceMouse request provenance required")
    if (provenance.get("sft_ready") is not False
            or provenance.get("robot_execution_approved") is not False
            or provenance.get("gripper_unknown_imputed") is not False):
        raise ValueError("Unexpected provenance or imputed gripper labels")
    if (provenance.get("source_action_horizon") != 10
            or provenance.get("frames_exported") != info.get("total_frames")
            or provenance.get("source_episodes") != info.get("total_episodes")):
        raise ValueError("Provenance and LeRobot dimensions/counts disagree")
    features = info.get("features", {})
    for key, width in (("state", STATE_DIM), ("actions", ACTION_DIM)):
        if list(features.get(key, {}).get("shape", [])) != [width]:
            raise ValueError(f"Bad {key} feature shape")
    paths = sorted((dataset / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("No LeRobot Parquet files")
    import pyarrow.parquet as pq
    states, actions = [], []
    for path in paths:
        parquet = pq.ParquetFile(path)
        if not {"state", "actions"}.issubset(parquet.schema.names):
            raise ValueError(f"Missing state/actions columns in {path}")
        for batch in parquet.iter_batches(columns=["state", "actions"], batch_size=4096):
            s = np.asarray(batch.column("state").to_pylist(), dtype=np.float64)
            a = np.asarray(batch.column("actions").to_pylist(), dtype=np.float64)
            if s.size == 0:
                continue
            if s.ndim != 2 or s.shape[1] != STATE_DIM or a.shape != (len(s), ACTION_DIM):
                raise ValueError(f"Bad numerical feature dimension in {path}")
            states.append(s)
            actions.append(a)
    if not states:
        raise ValueError("No training samples")
    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    if len(state) != provenance["frames_exported"]:
        raise ValueError("Parquet row count differs from provenance")
    stats = compute_from_arrays(state, action)
    summary = {
        "schema": SCHEMA,
        "source_split": "train_only",
        "dataset_path": str(dataset),
        "dataset_provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "num_samples": len(state),
        "state_dim": STATE_DIM,
        "request_action_dim": ACTION_DIM,
        "model_padded_dim": MODEL_DIM,
        "normalization": "OpenPI Pi0.5 quantile q01/q99, padded inactive dimensions identity",
        "action_semantics": "normalized SpaceMouse request 6D + binary gripper open request",
        "validation_samples_used": 0,
        "controller_command_mapping_verified": False,
        "sft_ready": False,
        "recap_ready": False,
        "robot_execution_approved": False,
    }
    return stats, summary


def save_stats(stats: dict, summary: dict, output: Path) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "norm_stats.json").write_text(
        json.dumps({"norm_stats": stats}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "request_norm_provenance.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stats, summary = compute_from_lerobot(args.train_dataset)
    save_stats(stats, summary, args.output)
    print(f"Training-only norm stats: {args.output / 'norm_stats.json'}")
    print(f"Samples: {summary['num_samples']}; 19D state / 7D request action padded to 32D")
    print("NOT a verified controller-action model. No robot commands/training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
