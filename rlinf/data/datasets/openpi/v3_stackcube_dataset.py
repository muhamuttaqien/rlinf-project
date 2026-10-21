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

LeRobot v3 stores contiguous episodes across Parquet shards and concatenates
camera frames into MP4 files, with per-episode offsets in meta/episodes Parquet.
The older LeRobot dependency used by RLinf's OpenPI fork cannot load that format.

We intentionally *do not* modify dataset files or the installed LeRobot package.
Actions are absolute pd_joint_pos (8D); the final action is repeated when a
10-step action window extends past an episode boundary, matching LeRobot's
delta-timestamps clamping. The duplicated wrist camera is masked by our OpenPI
ManiSkillInputs transform and is therefore not decoded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import SupportsIndex

import numpy as np


def is_lerobot_v3_dataset(root: Path) -> bool:
    """Detect the Parquet-only metadata in datasets written by LeRobot >=0.4."""
    return (
        (root / "meta" / "info.json").is_file()
        and (root / "meta" / "tasks.parquet").is_file()
        and not (root / "meta" / "tasks.jsonl").is_file()
    )


class V3StackCubeDataset:
    """Map-style v3 dataset, returning the raw keys expected by OpenPI transforms."""

    def __init__(self, root: str | Path, *, action_horizon: int):
        import pyarrow.parquet as pq

        self.root = Path(root).expanduser().resolve()
        info_path = self.root / "meta" / "info.json"
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"Invalid LeRobot fps: {self.fps}")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self.action_horizon = action_horizon

        features = self.info["features"]
        for key, shape in (("state", [9]), ("actions", [8])):
            if list(features.get(key, {}).get("shape", [])) != shape:
                raise ValueError(
                    f"Expected StackCube {key} {shape}, got {features.get(key)}"
                )
        if "image" not in features:
            raise ValueError("StackCube v3 dataset is missing the base camera 'image'")

        data_files = sorted((self.root / "data").rglob("*.parquet"))
        episode_files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not data_files or not episode_files:
            raise FileNotFoundError(
                f"Expected data and meta/episodes Parquet shards under {self.root}"
            )

        fields = ("index", "episode_index", "frame_index", "timestamp", "state", "actions")
        frames = []
        for file in data_files:
            schema = set(pq.read_schema(file).names)
            missing = set(fields) - schema
            if missing:
                raise ValueError(f"{file} missing {sorted(missing)}")
            frames.extend(
                pq.read_table(file, columns=list(fields)).to_pylist()
            )
        frames.sort(key=lambda row: int(row["index"]))
        if len(frames) != int(self.info["total_frames"]):
            raise ValueError(
                f"Frame count mismatch: {len(frames)} != {self.info['total_frames']}"
            )
        indices = [int(row["index"]) for row in frames]
        if indices != list(range(len(frames))):
            raise ValueError("Frame indices must be unique and contiguous from zero")
        self.frames = frames

        episodes = {}
        for file in episode_files:
            for row in pq.read_table(file).to_pylist():
                index = int(row["episode_index"])
                if index in episodes:
                    raise ValueError(f"Duplicate episode metadata: {index}")
                episodes[index] = row
        if len(episodes) != int(self.info["total_episodes"]):
            raise ValueError(
                f"Episode count mismatch: {len(episodes)} != {self.info['total_episodes']}"
            )

        self.episodes = episodes
        self.last_indices = {}
        cursor = 0
        for ep_id in sorted(episodes):
            ep = episodes[ep_id]
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
            if start != cursor or end <= start or end > len(frames):
                raise ValueError(
                    f"Episode {ep_id} has invalid frame bounds [{start}, {end})"
                )
            if int(ep["length"]) != end - start:
                raise ValueError(f"Episode {ep_id} length differs from bounds")
            for offset, row in enumerate(frames[start:end]):
                if int(row["episode_index"]) != ep_id or int(row["frame_index"]) != offset:
                    raise ValueError(
                        f"Episode {ep_id} frame {offset} does not match metadata"
                    )
            for index in range(start, end):
                self.last_indices[index] = end - 1
            cursor = end
        if cursor != len(frames):
            raise ValueError("Episode frame bounds do not cover the full dataset")

        self.video_path = self.info.get("video_path")
        if not isinstance(self.video_path, str):
            raise ValueError(
                "LeRobot v3 StackCube must have MP4 video_path in meta/info.json"
            )
        for ep_id, episode in self.episodes.items():
            for key in (
                "videos/image/chunk_index",
                "videos/image/file_index",
                "videos/image/from_timestamp",
            ):
                if key not in episode:
                    raise ValueError(f"Episode {ep_id} missing {key}")

    def __len__(self) -> int:
        return len(self.frames)

    def _read_image(self, frame: dict) -> np.ndarray:
        """Decode the base-camera frame by its v3 video offset and local timestamp."""
        import cv2

        ep_id = int(frame["episode_index"])
        ep = self.episodes[ep_id]
        relative_path = self.video_path.format(
            video_key="image",
            chunk_index=int(ep["videos/image/chunk_index"]),
            file_index=int(ep["videos/image/file_index"]),
        )
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise FileNotFoundError(f"LeRobot v3 base camera video not found: {path}")

        timestamp = float(ep["videos/image/from_timestamp"]) + float(
            frame["timestamp"]
        )
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open LeRobot v3 video: {path}")
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            success, bgr = capture.read()
        finally:
            capture.release()
        if not success or bgr is None:
            raise RuntimeError(
                f"Unable to decode base-camera frame at {timestamp:.5f}s in {path}"
            )
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_shape = list(self.info["features"]["image"]["shape"])
        if list(image.shape) != image_shape:
            raise ValueError(
                f"Incorrect decoded frame shape {image.shape}, expected {image_shape}"
            )
        return image

    def __getitem__(self, idx: SupportsIndex) -> dict:
        idx = idx.__index__()
        if idx < 0 or idx >= len(self.frames):
            raise IndexError(idx)
        frame = self.frames[idx]
        last = self.last_indices[idx]
        actions = np.asarray(
            [
                self.frames[min(idx + offset, last)]["actions"]
                for offset in range(self.action_horizon)
            ],
            dtype=np.float32,
        )
        state = np.asarray(frame["state"], dtype=np.float32)
        if actions.shape != (self.action_horizon, 8) or state.shape != (9,):
            raise ValueError(
                f"Invalid sample {idx}: state {state.shape}, actions {actions.shape}"
            )
        if not np.isfinite(actions).all() or not np.isfinite(state).all():
            raise ValueError(f"Nonfinite numerical sample {idx}")
        image = self._read_image(frame)
        return {
            "state": state,
            "actions": actions,
            "image": image,
            "wrist_image": np.zeros_like(image),
            "episode_index": np.int64(frame["episode_index"]),
            "frame_index": np.int64(frame["frame_index"]),
        }
