# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline pi0.5 2-camera dataset, normalization and official loader preflight.

NEVER trains, runs inference, accesses ROS or commands an FR3. Requires
independent train/validation request-label splits and TRAIN-ONLY stats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from toolkits.realworld_check.fr3_request_lerobot_export import EXPORT_SCHEMA
from toolkits.realworld_check.fr3_request_norm_stats import SCHEMA as NORM_SCHEMA


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def validate_metadata(train: Path, validation: Path, norm: Path) -> dict:
    train = train.expanduser().resolve()
    validation = validation.expanduser().resolve()
    norm = norm.expanduser().resolve()
    if train == validation or norm.is_relative_to(validation):
        raise ValueError("Train/validation leakage risk")
    summaries = {}
    ids = {}
    for name, path in (("train", train), ("validation", validation)):
        p = _json(path / "fr3_command_request_provenance.json")
        info = _json(path / "meta" / "info.json")
        if (p.get("schema") != EXPORT_SCHEMA or p.get("source_split") != name
                or p.get("sft_ready") is not False
                or p.get("robot_execution_approved") is not False
                or p.get("frames_exported") != info.get("total_frames")
                or p.get("source_action_horizon") != 10):
            raise ValueError(f"{name}: missing/mismatched request provenance")
        if info.get("total_episodes") != p.get("source_episodes"):
            raise ValueError(f"{name}: episode count inconsistent")
        features = info.get("features", {})
        for key, expected in (("state", [19]), ("actions", [7])):
            if features.get(key, {}).get("shape") != expected:
                raise ValueError(f"{name}: wrong {key} dimensions")
        for key in ("image", "extra_view_image"):
            shape = features.get(key, {}).get("shape")
            if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3:
                raise ValueError(f"{name}: missing RGB camera {key}")
        ids[name] = {
            x["episode"] for x in p.get("segments", [])
        }
        summaries[name] = {
            "episodes": info["total_episodes"],
            "rows": info["total_frames"],
            "source_episodes": len(ids[name]),
        }
    if not ids["train"].isdisjoint(ids["validation"]):
        raise ValueError("Train/validation source episode overlap")
    if summaries["train"]["source_episodes"] != 80 \
            or summaries["validation"]["source_episodes"] != 20:
        raise ValueError("Expected 80 train and 20 validation source episodes")

    stats_file = norm / "norm_stats.json"
    stats = _json(stats_file).get("norm_stats", {})
    metadata = _json(norm / "request_norm_provenance.json")
    train_digest = hashlib.sha256(
        (train / "fr3_command_request_provenance.json").read_bytes()
    ).hexdigest()
    if (metadata.get("schema") != NORM_SCHEMA
            or metadata.get("source_split") != "train_only"
            or metadata.get("dataset_path") != str(train)
            or metadata.get("dataset_provenance_sha256") != train_digest
            or metadata.get("validation_samples_used") != 0
            or metadata.get("num_samples") != summaries["train"]["rows"]):
        raise ValueError("Stats not exclusively generated from this training dataset")
    for key in ("state", "actions"):
        item = stats.get(key, {})
        for field in ("mean", "std", "q01", "q99"):
            arr = np.asarray(item.get(field), dtype=np.float64)
            if arr.shape != (32,) or not np.isfinite(arr).all():
                raise ValueError(f"Bad {key} {field} normalization")
        if np.any(np.asarray(item["q99"]) <= np.asarray(item["q01"])):
            raise ValueError(f"Invalid {key} quantile range")
    return {
        **summaries,
        "norm_file": str(stats_file),
        "train_only_norm_passed": True,
        "source_episode_disjoint": True,
        "sft_ready": False,
        "robot_execution_approved": False,
    }


def check_official_loader(train: Path, validation: Path, norm: Path,
                          model: Path) -> dict:
    report = validate_metadata(train, validation, norm)
    from omegaconf import OmegaConf
    from rlinf.data.datasets.openpi.official_sft_data_loader import (
        build_official_openpi_sft_dataloader,
    )
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    name = "pi05_fr3_peg_request_offline"
    config = get_openpi_config(
        name, model_path=str(model), repo_id=str(train),
        data_kwargs={"norm_stats_path": str(norm / "norm_stats.json")},
    )
    if config.model.action_horizon != 10 or config.model.action_dim != 32:
        raise ValueError("Wrong OpenPI request model shape")
    summary = {}
    for split, path in (("train", train), ("validation", validation)):
        cfg = OmegaConf.create({
            "data": {"num_workers": 0},
            "actor": {
                "seed": 0, "micro_batch_size": 1, "eval_batch_size": 1,
                "model": {
                    "model_path": str(model),
                    "openpi": {
                        "config_name": name,
                        "model_action_dim": 32,
                        "action_horizon": 10,
                    },
                    "openpi_data": {
                        "norm_stats_path": str(norm / "norm_stats.json"),
                    },
                },
            },
        })
        loader, data_cfg = build_official_openpi_sft_dataloader(
            cfg, world_size=1, rank=0, data_paths=str(path),
            eval_dataset=(split == "validation"),
        )
        if data_cfg.norm_stats is None:
            raise ValueError("Missing OpenPI norm stats")
        expected_starts = (
            11568 if split == "train" else 2898
        )
        if getattr(loader, "strict_valid_starts_count", None) != expected_starts:
            raise ValueError(
                f"{split}: strict horizon count differs from audit: "
                f"{getattr(loader, 'strict_valid_starts_count', None)} "
                f"!= {expected_starts}"
            )
        sample = next(iter(loader))
        if not isinstance(sample, (list, tuple)) or len(sample) != 2:
            raise ValueError("Unexpected official loader response")
        actions = np.asarray(sample[1])
        if actions.shape[-2:] != (10, 32) or not np.isfinite(actions).all():
            raise ValueError(f"{split}: invalid normalized action batch")
        observation = sample[0]
        images = observation.images if hasattr(observation, "images") else None
        if not isinstance(images, dict) or not {"base_0_rgb", "left_wrist_0_rgb"}.issubset(images):
            raise ValueError(f"{split}: two-camera observation unavailable")
        summary[split] = {
            "actions_shape": list(actions.shape),
            "image_keys": sorted(images),
            "official_loader_passed": True,
            "strict_valid_horizon_starts": expected_starts,
            "boundary_padding_excluded": True,
        }
    return {**report, "loader_results": summary, "model_trained": False}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--validation", required=True, type=Path)
    p.add_argument("--norm-dir", required=True, type=Path)
    p.add_argument("--model-path", type=Path)
    args = p.parse_args()
    result = (
        validate_metadata(args.train, args.validation, args.norm_dir)
        if args.model_path is None
        else check_official_loader(
            args.train, args.validation, args.norm_dir, args.model_path
        )
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    print("OFFLINE preflight only; no SFT/RECAP or robot control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
