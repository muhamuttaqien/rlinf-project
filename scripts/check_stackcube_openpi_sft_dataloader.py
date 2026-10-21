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

"""Read one real LeRobot v3 StackCube batch through the official OpenPI SFT pipeline.

This script DOES NOT load checkpoint weights or launch GPU training.
Run from the repository root:
    PYTHONPATH="$PWD" python scripts/check_stackcube_openpi_sft_dataloader.py
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import torch

from rlinf.data.datasets.openpi.official_sft_data_loader import (
    _create_openpi_v3_data_loader,
)
from rlinf.data.datasets.openpi.v3_stackcube_dataset import (
    V3StackCubeDataset,
    is_lerobot_v3_dataset,
)
from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/stackcube_sft_pilot100")
    parser.add_argument("--norm-stats", default=os.getenv("STACKCUBE_NORM_STATS"))
    args = parser.parse_args()

    root = resolve_lerobot_dataset_root(args.repo_id)
    if not is_lerobot_v3_dataset(root):
        raise ValueError(f"Expected LeRobot v3 dataset at {root}")

    norm_file = Path(
        args.norm_stats
        or "assets/pi05_stackcube_panda_joint/local/stackcube_sft_pilot100/norm_stats.json"
    ).expanduser().resolve()
    if not norm_file.is_file():
        raise FileNotFoundError(f"Normalization file missing: {norm_file}")

    raw = V3StackCubeDataset(root, action_horizon=10)
    print("Dataset:", root)
    print("Frames:", len(raw), "Episodes:", len(raw.episodes))
    for idx in (0, len(raw) // 2, len(raw) - 1):
        sample = raw[idx]
        assert sample["state"].shape == (9,)
        assert sample["actions"].shape == (10, 8)
        assert sample["image"].shape == (224, 224, 3)
        assert not sample["wrist_image"].any()
    print("Three raw v3 samples: OK")

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

    data_loader = _create_openpi_v3_data_loader(
        loader, config, root, shuffle=False
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
    print("StackCube OpenPI v3 SFT dataloader passed")


if __name__ == "__main__":
    main()
