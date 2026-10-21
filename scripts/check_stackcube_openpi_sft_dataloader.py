# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read one StackCube batch through the OpenPI SFT pipeline (LeRobot v2/v3).

This script DOES NOT load checkpoint weights or launch GPU training.
Run from the repository root:
    PYTHONPATH="$PWD" python scripts/check_stackcube_openpi_sft_dataloader.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch

from rlinf.data.datasets.openpi.official_sft_data_loader import (
    _create_openpi_v3_data_loader,
)
from rlinf.data.datasets.openpi.v3_stackcube_dataset import (
    V3StackCubeDataset,
)
from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def detect_lerobot_metadata_format(root: Path) -> str:
    """Identify the actual local metadata format; never guess or query the Hub."""
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"StackCube dataset manifest missing: {info_path}. "
            "Collect the demonstrations first, or correct --repo-id."
        )

    v3 = (root / "meta" / "tasks.parquet").is_file()
    v2 = (root / "meta" / "tasks.jsonl").is_file()
    if v3 and not v2:
        return "v3"
    if v2 and not v3:
        return "v2"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    raise ValueError(
        f"Unrecognized/incomplete LeRobot task metadata at {root}: "
        f"codebase_version={info.get('codebase_version')!r}, "
        f"tasks.parquet={v3}, tasks.jsonl={v2}. "
        "Keep the dataset unchanged and inspect its meta/ directory."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/stackcube_sft_pilot100")
    parser.add_argument("--norm-stats", default=os.getenv("STACKCUBE_NORM_STATS"))
    args = parser.parse_args()

    root = resolve_lerobot_dataset_root(args.repo_id)
    metadata_format = detect_lerobot_metadata_format(root)
    print(f"Dataset: {root}")
    print(f"LeRobot metadata: {metadata_format}")

    norm_file = Path(
        args.norm_stats
        or "assets/pi05_stackcube_panda_joint/local/stackcube_sft_pilot100/norm_stats.json"
    ).expanduser().resolve()
    if not norm_file.is_file():
        raise FileNotFoundError(f"Normalization file missing: {norm_file}")

    if metadata_format == "v3":
        raw = V3StackCubeDataset(root, action_horizon=10)
    else:
        # Older LeRobot datasets use meta/tasks.jsonl. The installed OpenPI
        # loader can read them natively, including episode-clamped actions.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        fps = float(info["fps"])
        raw = LeRobotDataset(
            args.repo_id,
            root=root,
            delta_timestamps={"actions": [i / fps for i in range(10)]},
        )
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert len(raw) == int(info["total_frames"])
    print("Frames:", len(raw), "Episodes:", info["total_episodes"])
    for idx in (0, len(raw) // 2, len(raw) - 1):
        sample = raw[idx]
        assert np.asarray(sample["state"]).shape == (9,)
        assert np.asarray(sample["actions"]).shape == (10, 8)
        assert np.asarray(sample["image"]).shape in (
            (3, 224, 224),
            (224, 224, 3),
        )
        # An original v2 collector may store a nonzero wrist image. OpenPI
        # masks it, so it must not contribute to the learning inputs.
        if metadata_format == "v3":
            assert not np.asarray(sample["wrist_image"]).any()
    print(f"Three raw {metadata_format} StackCube samples: OK")

    config = get_openpi_config(
        "pi05_stackcube_panda_joint",
        repo_id=args.repo_id,
        batch_size=1,
        data_kwargs={"norm_stats_path":str(norm_file)},
    )
    config = dataclasses.replace(config, num_workers=0)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise RuntimeError(f"OpenPI failed to load norm stats from {norm_file}")

    import openpi.training.data_loader as loader

    if metadata_format == "v3":
        data_loader = _create_openpi_v3_data_loader(
            loader, config, root, shuffle=False
        )
    else:
        data_loader = loader.create_data_loader(
            config, framework="pytorch", shuffle=False
        )
    obs, actions = next(iter(data_loader))
    assert actions.shape == (1, 10, 32), actions.shape
    assert obs.state.shape == (1, 32), obs.state.shape
    assert obs.image_masks["base_0_rgb"].all()
    assert not obs.image_masks["left_wrist_0_rgb"].any()
    assert not obs.image_masks["right_wrist_0_rgb"].any()
    assert torch.isfinite(actions).all()
    assert torch.isfinite(obs.state).all()
    assert obs.tokenized_prompt is not None

    print("Normalized OpenPI SFT batch:", tuple(actions.shape))
    print("Padded model state:", tuple(obs.state.shape))
    print("Base-camera image:", tuple(obs.images["base_0_rgb"].shape))
    print("Wrist camera: masked")
    print(f"StackCube OpenPI {metadata_format} SFT dataloader passed")


if __name__ == "__main__":
    main()
