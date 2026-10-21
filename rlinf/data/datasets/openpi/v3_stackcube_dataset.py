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

"""Read-only LeRobot v3 StackCube samples for the pinned OpenPI SFT loader.

The original StackCube collector stores images as LeRobot 'image' features
embedded in Parquet, NOT 'video' features backed by MP4. Reuse the installed
LeRobot v3 reader for both media decoding and episode-safe action windows.
OpenPI's own training loader expects the older v2.1 metadata representation;
this adapter changes only the source dataset while retaining official OpenPI
repacking, quantile normalization, tokenization and action padding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import SupportsIndex

import numpy as np


def is_lerobot_v3_dataset(root: Path) -> bool:
    """Detect Parquet task metadata written by LeRobot >= 0.4."""
    return (
        (root / "meta" / "info.json").is_file()
        and (root / "meta" / "tasks.parquet").is_file()
        and not (root / "meta" / "tasks.jsonl").is_file()
    )


class V3StackCubeDataset:
    """Read one base-camera image and a 10-step episode-safe action window."""

    def __init__(
        self, root: str | Path, *, action_horizon: int, repo_id: str | None = None
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root).expanduser().resolve()
        self.info = json.loads(
            (self.root / "meta" / "info.json").read_text(encoding="utf-8")
        )
        if not is_lerobot_v3_dataset(self.root):
            raise ValueError(f"Not a LeRobot v3 dataset: {self.root}")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self.action_horizon = action_horizon

        features = self.info.get("features", {})
        for key, shape in (("state", [9]), ("actions", [8])):
            if list(features.get(key, {}).get("shape", [])) != shape:
                raise ValueError(
                    f"Expected StackCube {key} {shape}, got {features.get(key)}"
                )
        camera = features.get("image", {})
        if list(camera.get("shape", [])) != [224, 224, 3]:
            raise ValueError(
                f"Expected 224x224x3 StackCube base image, got {camera}"
            )
        if camera.get("dtype") not in ("image", "video"):
            raise ValueError(
                f"Expected LeRobot image/video camera feature, got {camera}"
            )

        fps = float(self.info["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid LeRobot fps: {fps}")

        self.dataset = LeRobotDataset(
            repo_id=repo_id or f"local/{self.root.name}",
            root=self.root,
            delta_timestamps={
                "actions": [step / fps for step in range(action_horizon)]
            },
        )
        if len(self.dataset) != int(self.info["total_frames"]):
            raise ValueError(
                "Frame count mismatch: native LeRobot loader returned "
                f"{len(self.dataset)} frames; info.json reports "
                f"{self.info['total_frames']}"
            )
        self.episodes = self.dataset.meta.episodes
        if len(self.episodes) != int(self.info["total_episodes"]):
            raise ValueError(
                "Episode count mismatch: native LeRobot loader returned "
                f"{len(self.episodes)} episodes; info.json reports "
                f"{self.info['total_episodes']}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: SupportsIndex) -> dict:
        idx = idx.__index__()
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        sample = self.dataset[idx]
        state = np.asarray(sample["state"], dtype=np.float32)
        actions = np.asarray(sample["actions"], dtype=np.float32)
        if state.shape != (9,) or actions.shape != (self.action_horizon, 8):
            raise ValueError(
                f"Invalid sample {idx}: state {state.shape}, actions {actions.shape}"
            )
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite numerical sample {idx}")

        # LeRobot v3 decodes embedded images as CHW float tensors. Keep them
        # unchanged for the ManiSkillInputs transform, which handles CHW->HWC
        # and 0..1 floats -> uint8. Do not waste resources decoding the fake
        # wrist camera in the policy: its mask remains False.
        image = np.asarray(sample["image"])
        if image.ndim != 3 or (
            image.shape != (3, 224, 224)
            and image.shape != (224, 224, 3)
        ):
            raise ValueError(f"Invalid base image shape at frame {idx}: {image.shape}")
        if not np.isfinite(image).all():
            raise ValueError(f"Nonfinite base image at frame {idx}")

        return {
            "state": state,
            "actions": actions,
            "image": image,
            "wrist_image": np.zeros_like(image),
            "episode_index": np.int64(sample["episode_index"]),
            "frame_index": np.int64(sample["frame_index"]),
        }
