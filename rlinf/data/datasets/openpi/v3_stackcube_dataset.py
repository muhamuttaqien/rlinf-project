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

"""Read-only StackCube LeRobot v3 adapter for RLinf's pinned OpenPI SFT loader.

The installed LeRobot reader is v2.1-only (requires meta/tasks.jsonl), whereas
StackCube has LeRobot v3 Parquet task/episode metadata and images embedded in
Parquet. Read the v3 files directly without updating the dataset, installing a
different LeRobot release, or changing OpenPI's normal transforms.

The numeric index is held in RAM, while embedded image bytes are loaded lazily
from a single Parquet row group. Actions are clamped to the final frame of the
same episode when the 10-step prediction window extends beyond its boundary.
"""

from __future__ import annotations

import bisect
import io
import json
from pathlib import Path
from typing import SupportsIndex

import numpy as np


def is_lerobot_v3_dataset(root: Path) -> bool:
    """Recognize the v3 task metadata layout, without Hub access."""
    return (
        (root / "meta" / "info.json").is_file()
        and (root / "meta" / "tasks.parquet").is_file()
        and not (root / "meta" / "tasks.jsonl").is_file()
    )


class V3StackCubeDataset:
    """Map-style samples compatible with OpenPI's ManiSkill repack transform."""

    def __init__(
        self, root: str | Path, *, action_horizon: int, repo_id: str | None = None,
        use_wrist_image: bool = False
    ):
        import pyarrow.parquet as pq

        del repo_id  # Local paths are resolved explicitly; never query Hugging Face.
        self.root = Path(root).expanduser().resolve()
        if not is_lerobot_v3_dataset(self.root):
            raise ValueError(f"Not a LeRobot v3 dataset: {self.root}")
        self.info = json.loads(
            (self.root / "meta" / "info.json").read_text(encoding="utf-8")
        )
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        self.action_horizon = action_horizon
        self.use_wrist_image = bool(use_wrist_image)

        features = self.info["features"]
        for key, shape in (("state", [9]), ("actions", [8])):
            if list(features.get(key, {}).get("shape", [])) != shape:
                raise ValueError(
                    f"Expected StackCube {key} {shape}, got {features.get(key)}"
                )
        camera = features.get("image", {})
        if camera.get("dtype") != "image" or list(camera.get("shape", [])) != [
            224, 224, 3
        ]:
            raise ValueError(
                "StackCube adapter expects an embedded 224x224 RGB image "
                f"feature, got {camera}"
            )

        if self.use_wrist_image:
            wrist = features.get("wrist_image", {})
            if wrist.get("dtype") != "image" or list(wrist.get("shape", [])) != [
                224, 224, 3
            ]:
                raise ValueError(
                    "Two-camera StackCube requires a real 224x224 wrist_image feature"
                )

        data_files = sorted((self.root / "data").rglob("*.parquet"))
        episode_files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not data_files or not episode_files:
            raise FileNotFoundError(
                f"Missing v3 data/episode Parquet files in {self.root}"
            )

        numeric_keys = (
            "index", "episode_index", "frame_index", "state", "actions"
        )
        frames = []
        for file in data_files:
            schema = set(pq.read_schema(file).names)
            image_keys = {"image", "wrist_image"} if self.use_wrist_image else {"image"}
            missing = (set(numeric_keys) | image_keys) - schema
            if missing:
                raise ValueError(f"{file} missing Parquet columns {sorted(missing)}")
            rows = pq.read_table(file, columns=list(numeric_keys)).to_pylist()
            for local_row, row in enumerate(rows):
                row["_file"] = str(file)
                row["_local_row"] = local_row
            frames.extend(rows)

        frames.sort(key=lambda row: int(row["index"]))
        expected_frames = int(self.info["total_frames"])
        if len(frames) != expected_frames:
            raise ValueError(
                f"Frame count mismatch: read {len(frames)}, expected {expected_frames}"
            )
        if [int(frame["index"]) for frame in frames] != list(
            range(expected_frames)
        ):
            raise ValueError("Frame indices must be unique and contiguous from zero")
        self.frames = frames

        episodes = {}
        for file in episode_files:
            for row in pq.read_table(file).to_pylist():
                ep_id = int(row["episode_index"])
                if ep_id in episodes:
                    raise ValueError(f"Duplicate episode metadata: {ep_id}")
                episodes[ep_id] = row
        expected_episodes = int(self.info["total_episodes"])
        if set(episodes) != set(range(expected_episodes)):
            raise ValueError(
                "Episode metadata indices must be contiguous from zero; "
                f"found {sorted(episodes)}"
            )
        self.episodes = episodes

        self._last_index = np.empty(len(frames), dtype=np.int64)
        cursor = 0
        for ep_id in sorted(episodes):
            ep = episodes[ep_id]
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
            if start != cursor or not start < end <= len(frames):
                raise ValueError(
                    f"Episode {ep_id} has inconsistent bounds [{start}, {end})"
                )
            if int(ep["length"]) != end - start:
                raise ValueError(f"Episode {ep_id} length differs from bounds")
            for offset, frame in enumerate(frames[start:end]):
                if (
                    int(frame["episode_index"]) != ep_id
                    or int(frame["frame_index"]) != offset
                ):
                    raise ValueError(
                        f"Episode {ep_id} frame {offset} differs from Parquet"
                    )
            self._last_index[start:end] = end - 1
            cursor = end
        if cursor != len(frames):
            raise ValueError("Episode bounds do not cover all frames")

        # Lazy per-worker cache; neither PyArrow files nor image bytes are
        # included in the pickled Dataset passed to DataLoader spawn workers.
        self._cached_group_key = None
        self._cached_group = None

    def __len__(self) -> int:
        return len(self.frames)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cached_group_key"] = None
        state["_cached_group"] = None
        return state

    def _embedded_image(self, frame: dict, *, image_key: str = "image") -> np.ndarray:
        """Decode the selected embedded RGB image from its Parquet row group."""
        import pyarrow.parquet as pq
        from PIL import Image

        path = Path(frame["_file"])
        row_offset = int(frame["_local_row"])
        parquet = pq.ParquetFile(path)
        offsets = [0]
        for i in range(parquet.num_row_groups):
            offsets.append(offsets[-1] + parquet.metadata.row_group(i).num_rows)
        group_index = bisect.bisect_right(offsets, row_offset) - 1
        if not 0 <= group_index < parquet.num_row_groups:
            raise IndexError(f"Parquet row offset outside file: {row_offset}")

        group_key = (str(path), group_index, image_key)
        if self._cached_group_key != group_key:
            self._cached_group = parquet.read_row_group(
                group_index, columns=[image_key]
            ).column(image_key)
            self._cached_group_key = group_key
        entry = self._cached_group[row_offset - offsets[group_index]].as_py()
        if not isinstance(entry, dict):
            raise ValueError(
                f"Expected embedded LeRobot {image_key} struct, got {type(entry)}"
            )
        encoded = entry.get("bytes")
        if encoded is not None:
            image_source = io.BytesIO(encoded)
        elif entry.get("path"):
            image_path = (self.root / str(entry["path"])).resolve()
            if not image_path.is_relative_to(self.root):
                raise ValueError(f"Image path escapes dataset root: {image_path}")
            image_source = image_path
        else:
            raise ValueError(
                "LeRobot image has neither bytes nor path: "
                f"frame {frame['index']}"
            )

        with Image.open(image_source) as image:
            array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        if array.shape != (224, 224, 3):
            raise ValueError(
                f"Invalid RGB image shape at frame {frame['index']}: {array.shape}"
            )
        return array

    def __getitem__(self, idx: SupportsIndex) -> dict:
        idx = idx.__index__()
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        frame = self.frames[idx]
        final_index = int(self._last_index[idx])
        actions = np.asarray(
            [
                self.frames[min(idx + offset, final_index)]["actions"]
                for offset in range(self.action_horizon)
            ],
            dtype=np.float32,
        )
        state = np.asarray(frame["state"], dtype=np.float32)
        if state.shape != (9,) or actions.shape != (self.action_horizon, 8):
            raise ValueError(
                f"Invalid sample {idx}: state {state.shape}, actions {actions.shape}"
            )
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite robot values at frame {idx}")

        image = self._embedded_image(frame)
        wrist_image = (
            self._embedded_image(frame, image_key="wrist_image")
            if self.use_wrist_image else np.zeros_like(image)
        )
        if self.use_wrist_image and np.array_equal(image, wrist_image):
            raise ValueError("Two-camera sample has identical base and wrist images")
        return {
            "state": state,
            "actions": actions,
            "image": image,
            # The legacy single-camera config still masks out this placeholder.
            "wrist_image": wrist_image,
            "episode_index": np.int64(frame["episode_index"]),
            "frame_index": np.int64(frame["frame_index"]),
        }
