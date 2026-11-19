# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline check of a FUTURE real FR3 LeRobot -> official pi0.5 SFT loader.

Does not train, run an inference policy, talk to ROS or deploy a controller.
Requires real labels, FR3-only 32D padded norm stats and a local dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from toolkits.realworld_check.fr3_labeled_episode_export import ACTION_SOURCE
from toolkits.realworld_check.franka_pi05_contract import validate_realworld_pi05_sample
from toolkits.realworld_check.fr3_dataset_setup_smoke import _rgb_image


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON object required: " + str(path))
    return data


def inspect_metadata(dataset: Path, norm_stats: Path) -> dict:
    """Fail closed for missing/incompatible labeled dataset or norm stats."""
    dataset = Path(dataset)
    info = _read_json(dataset / "meta" / "info.json")
    provenance = _read_json(dataset / "fr3_label_provenance.json")
    if provenance.get("label_source") != ACTION_SOURCE:
        raise ValueError("Dataset lacks FR3 controller-accepted action provenance")
    if provenance.get("physical_policy_execution_approved") is not False:
        raise ValueError("Export must not claim physical policy approval")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("LeRobot features missing")
    for key, size in (("state", 19), ("actions", 7)):
        item = features.get(key)
        if not isinstance(item, dict) or list(item.get("shape", [])) != [size]:
            raise ValueError("Expected FR3 19D state and 7D action features")
    for key in ("image", "extra_view_image"):
        item = features.get(key)
        shape = item.get("shape") if isinstance(item, dict) else None
        if (not isinstance(shape, (tuple, list)) or len(shape) != 3
                or shape[-1] != 3):
            raise ValueError("Both real RGB camera features are mandatory")
    if info.get("total_frames", 0) < 1 or info.get("total_episodes", 0) < 1:
        raise ValueError("No labeled episodes in dataset")

    norm_stats = Path(norm_stats)
    if norm_stats.name != "norm_stats.json" or not norm_stats.is_file():
        raise FileNotFoundError("Explicit FR3 norm_stats.json required")
    payload = _read_json(norm_stats)
    payload = payload.get("norm_stats", payload)
    dims = {}
    for key in ("state", "actions"):
        stats = payload.get(key)
        if not isinstance(stats, dict) or "mean" not in stats:
            raise ValueError("Missing normalization statistics for " + key)
        vec = np.asarray(stats["mean"], dtype=np.float64)
        if vec.shape != (32,) or not np.isfinite(vec).all():
            raise ValueError(key + " must be padded to OpenPI model dim 32")
        dims[key] = 32
    return {
        "dataset_path": str(dataset.resolve()),
        "total_frames": info["total_frames"],
        "total_episodes": info["total_episodes"],
        "pi05_model_action_dim": 32,
        "fr3_norm_dimensions": dims,
        "physical_execution_approved": False,
    }


def check_official_loader(dataset: Path, norm_stats: Path, model_path: Path) -> dict:
    """Load one actual LeRobot sample and a single official OpenPI SFT batch."""
    metadata = inspect_metadata(dataset, norm_stats)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from omegaconf import OmegaConf
    from openpi.models.model import ModelType
    from rlinf.data.datasets.openpi.official_sft_data_loader import (
        build_official_openpi_sft_dataloader,
    )
    from rlinf.models.embodiment.openpi.policies.realworld_policy import RealworldInputs
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    repo_id = _read_json(dataset / "fr3_label_provenance.json")["repo_id"]
    ds = LeRobotDataset(repo_id, root=dataset, download_videos=False)
    row = ds[0]
    sample = {
        "observation/image": _rgb_image(row["image"]),
        "observation/extra_view_image": _rgb_image(row["extra_view_image"]),
        "observation/state": np.asarray(row["state"], dtype=np.float32),
        "actions": np.asarray(row["actions"], dtype=np.float32).reshape(1, 7),
        "prompt": "offline realworld transform preflight",
    }
    validate_realworld_pi05_sample(sample)
    result = RealworldInputs(
        action_dim=32, model_type=ModelType.PI05
    )(sample)
    if np.asarray(result["state"]).shape != (32,):
        raise ValueError("OpenPI state not padded to 32")
    if tuple(bool(result["image_mask"][key]) for key in (
        "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
    )) != (True, True, False):
        raise ValueError("Expected two real camera views and one masked slot")

    # Reuse the identical production SFT loader builder, not a mock dataloader.
    cfg = OmegaConf.create({
        "data": {"num_workers": 0},
        "actor": {
            "seed": 0, "micro_batch_size": 1,
            "model": {
                "model_path": str(model_path),
                "openpi": {
                    "config_name": "pi05_franka_pnp",
                    "model_action_dim": 32,
                    "action_horizon": 5,
                },
                "openpi_data": {"norm_stats_path": str(norm_stats)},
            },
        },
    })
    loader, _ = build_official_openpi_sft_dataloader(
        cfg, world_size=1, rank=0, data_paths=str(dataset), eval_dataset=False
    )
    batch = next(iter(loader))
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Official OpenPI SFT loader returned unexpected batch")
    actions = batch[1]
    if actions.shape[-2:] != (5, 32):
        raise ValueError("Official pi0.5 action horizon / model dim mismatch")

    return {
        **metadata,
        "real_le_robot_sample_passed": True,
        "official_two_camera_transform_passed": True,
        "official_openpi_sft_batch_shape": list(actions.shape),
        "official_openpi_sft_loader_passed": True,
        "model_trained": False,
        "robot_commands_sent": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    args = parser.parse_args()
    report = check_official_loader(args.dataset, args.norm_stats, args.model_path)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
