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

"""Audit StackCube LeRobot data before RECAP preprocessing or model training.

The fast audit reads only Parquet state/action/label columns and metadata.
Use --decode-frames 3 to additionally check sampled images via LeRobot.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

STATE_DIM = 9
ACTION_DIM = 8
REQUIRED_FEATURES = ("image", "wrist_image", "state", "actions", "is_success")


def scalar_bool(value: object) -> bool:
    """Validate a Boolean label stored as bool or a single-element list."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            raise ValueError(f"is_success must have one value, found {value!r}")
        value = value[0]
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"is_success must be Boolean, found {value!r}")
    return bool(value)


def validate_episode_rows(
    episodes: dict[int, list[dict]],
    *,
    expect_sft: bool = True,
) -> dict:
    """Check sequential frame indices and typed Panda observations/actions."""
    if not episodes:
        raise ValueError("No episode data found")
    if set(episodes) != set(range(len(episodes))):
        raise ValueError(f"Expected episode indices 0..{len(episodes) - 1}, got {sorted(episodes)}")

    lengths = []
    success_count = 0
    for ep_idx in sorted(episodes):
        rows = sorted(episodes[ep_idx], key=lambda row: int(row["frame_index"]))
        frames = [int(row["frame_index"]) for row in rows]
        if frames != list(range(len(rows))):
            raise ValueError(f"Episode {ep_idx}: frame_index values are not consecutive from 0")
        if not rows:
            raise ValueError(f"Episode {ep_idx}: empty episode")

        successes = set()
        for row in rows:
            state = np.asarray(row["state"], dtype=np.float32)
            action = np.asarray(row["actions"], dtype=np.float32)
            if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
                raise ValueError(f"Episode {ep_idx}, frame {row['frame_index']}: invalid state {state.shape}")
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError(f"Episode {ep_idx}, frame {row['frame_index']}: invalid action {action.shape}")
            successes.add(scalar_bool(row["is_success"]))
        if len(successes) != 1:
            raise ValueError(f"Episode {ep_idx}: inconsistent is_success labels")
        successful = successes.pop()
        if expect_sft and not successful:
            raise ValueError(f"Episode {ep_idx} is marked failed but SFT data must succeed")
        success_count += int(successful)
        lengths.append(len(rows))

    return {
        "episodes": len(episodes),
        "frames": sum(lengths),
        "successful_episodes": success_count,
        "failed_episodes": len(episodes) - success_count,
        "episode_steps_min": min(lengths),
        "episode_steps_max": max(lengths),
        "episode_steps_mean": round(float(np.mean(lengths)), 2),
    }


def _load_tasks(root: Path) -> dict[int, str]:
    """Read task names from either supported LeRobot metadata representation."""
    from rlinf.data.storage.lerobot.metadata import load_task_descriptions

    return load_task_descriptions(root)

def audit_dataset(root: Path, expected_episodes: int | None = None) -> dict:
    """Validate manifest, per-frame numerical data, labels, and task IDs."""
    import pyarrow.parquet as pq

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot manifest not found: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    missing = set(REQUIRED_FEATURES) - set(features)
    if missing:
        raise ValueError(f"Missing required dataset features: {sorted(missing)}")
    for name, shape in (("state", [STATE_DIM]), ("actions", [ACTION_DIM]), ("is_success", [1])):
        actual = list(features[name].get("shape", ()))
        if actual != shape:
            raise ValueError(f"Invalid feature shape {name}: expected {shape}, got {actual}")

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode Parquet files found under {root / 'data'}")

    tasks = _load_tasks(root)
    if not tasks:
        raise ValueError("Missing task descriptions in meta/tasks.jsonl or meta/tasks.parquet")

    episode_rows: dict[int, list[dict]] = defaultdict(list)
    for filename in files:
        available = set(pq.read_schema(filename).names)
        required = {"episode_index", "frame_index", "state", "actions", "is_success"}
        if not required.issubset(available):
            raise ValueError(f"{filename}: missing Parquet columns {sorted(required - available)}")
        columns = sorted(required | ({"task_index"} if "task_index" in available else {"task"} if "task" in available else set()))
        if "task_index" not in columns and "task" not in columns:
            raise ValueError(f"{filename}: no task or task_index field to identify the prompt")

        for row in pq.read_table(filename, columns=columns).to_pylist():
            if "task_index" in row:
                task_index = int(row["task_index"])
                if task_index not in tasks:
                    raise ValueError(f"{filename}: unknown task_index {task_index}")
            elif str(row["task"]) not in tasks.values():
                raise ValueError(f"{filename}: task description not found in metadata")
            episode_rows[int(row["episode_index"])].append(row)

    report = validate_episode_rows(episode_rows)
    report["dataset"] = str(root)
    report["parquet_files"] = len(files)
    report["tasks"] = list(tasks.values())
    report["manifest_total_episodes"] = info.get("total_episodes")
    report["manifest_total_frames"] = info.get("total_frames")
    if int(info.get("total_episodes", -1)) != report["episodes"]:
        raise ValueError(f"Manifest total_episodes disagrees with Parquet: {report}")
    if int(info.get("total_frames", -1)) != report["frames"]:
        raise ValueError(f"Manifest total_frames disagrees with Parquet: {report}")
    if expected_episodes is not None and report["episodes"] != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, found {report['episodes']}")
    return report


def decode_sampled_frames(root: Path, count: int, total_frames: int) -> None:
    """Confirm sampled media can be opened by RLinf's LeRobot data loader."""
    if not count:
        return
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(str(root), download_videos=False)
    if len(ds) != total_frames:
        raise ValueError(f"LeRobot sees {len(ds)} frames, expected {total_frames}")
    for index in np.linspace(0, len(ds) - 1, min(count, len(ds)), dtype=int):
        sample = ds[int(index)]
        for name in ("image", "wrist_image"):
            array = np.asarray(sample[name])
            if array.ndim != 3 or 3 not in (array.shape[0], array.shape[-1]):
                raise ValueError(f"Frame {index} {name}: expected RGB image, got {array.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--decode-frames", type=int, default=0)
    args = parser.parse_args()
    if args.decode_frames < 0:
        parser.error("--decode-frames must be nonnegative")
    root = args.dataset_path.expanduser().resolve()
    report = audit_dataset(root, expected_episodes=args.expected_episodes)
    decode_sampled_frames(root, args.decode_frames, report["frames"])
    report["decoded_frames"] = min(args.decode_frames, report["frames"])
    report["status"] = "passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
