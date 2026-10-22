# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One focused audit of persisted successful StackCube EE-delta LeRobot data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root


def _image_hwc(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.shape == (3, 224, 224):
        image = np.transpose(image, (1, 2, 0))
    if image.shape != (224, 224, 3):
        raise ValueError(f"Expected 224x224 RGB image, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite camera image")
    return image


def check_dataset(repo_id: str, expected_episodes: int) -> dict:
    from rlinf.data.datasets.openpi.v3_stackcube_dataset import (
        V3StackCubeDataset, is_lerobot_v3_dataset,
    )

    root = resolve_lerobot_dataset_root(repo_id)
    meta = root / "meta" / "info.json"
    manifest_file = root / "collection_manifest.json"
    if not meta.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(
            f"Missing LeRobot metadata or collection manifest in {root}"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    if len(episodes) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} successful manifest episodes, "
            f"got {len(episodes)}"
        )

    if is_lerobot_v3_dataset(root):
        dataset = V3StackCubeDataset(
            root, action_horizon=10, use_wrist_image=True,
            state_dim=8, action_dim=7,
        )
        records = dataset.frames
        states = np.asarray([r["state"] for r in records], dtype=np.float32)
        actions = np.asarray([r["actions"] for r in records], dtype=np.float32)
        episode_ids = np.asarray(
            [r["episode_index"] for r in records], dtype=np.int64,
        )
        frame_ids = np.asarray(
            [r["frame_index"] for r in records], dtype=np.int64,
        )
        dataset_format = "v3"
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=repo_id, root=root)
        records = dataset.hf_dataset
        states = np.asarray(records["state"], dtype=np.float32)
        actions = np.asarray(records["actions"], dtype=np.float32)
        episode_ids = np.asarray(records["episode_index"], dtype=np.int64)
        frame_ids = np.asarray(records["frame_index"], dtype=np.int64)
        dataset_format = "native-lerobot"

    if states.shape != (len(dataset), 8):
        raise ValueError(f"Unexpected 8D states: {states.shape}")
    if actions.shape != (len(dataset), 7):
        raise ValueError(f"Unexpected 7D actions: {actions.shape}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite states/actions")
    if np.any(np.abs(actions) > 1.05):
        raise ValueError("Expert controller actions exceed normalized range")
    if not (np.any(actions[:, 6] > 0.5) and np.any(actions[:, 6] < -0.5)):
        raise ValueError("Dataset is missing a full open/close gripper phase")

    total_steps = 0
    for expected_id, episode in enumerate(episodes):
        if int(episode["episode_index"]) != expected_id:
            raise ValueError(f"Noncontiguous manifest episode {episode}")
        mask = episode_ids == expected_id
        steps = int(episode["steps"])
        if mask.sum() != steps:
            raise ValueError(
                f"Episode {expected_id}: manifest={steps} frames, "
                f"dataset={mask.sum()}"
            )
        if not np.array_equal(frame_ids[mask], np.arange(steps)):
            raise ValueError(f"Episode {expected_id}: invalid frame indices")
        total_steps += steps

    if total_steps != len(dataset):
        raise ValueError("Manifest lengths do not cover complete LeRobot dataset")

    # Only decode three saved image pairs, proportional to the verification goal.
    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for i in indices:
        row = dataset[i]
        base = _image_hwc(row["image"])
        wrist = _image_hwc(row["wrist_image"])
        if np.array_equal(base, wrist):
            raise ValueError(f"Base and wrist views identical at frame {i}")

    return {
        "dataset": str(root),
        "format": dataset_format,
        "episodes": expected_episodes,
        "frames": len(dataset),
        "state_dim": 8,
        "action_dim": 7,
        "image_pairs_checked": len(indices),
        "gripper_min": float(actions[:, 6].min()),
        "gripper_max": float(actions[:, 6].max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-id", default="local/stackcube_pi05_clean_v1_acceptance",
    )
    p.add_argument("--episodes", type=int, default=1)
    args = p.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    result = check_dataset(args.repo_id, args.episodes)
    print("DATASET AUDIT PASSED " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
