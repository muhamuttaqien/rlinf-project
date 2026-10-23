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

"""Inspect stored observations across the first expert gripper-close transition.

Reads a few frames of an existing successful LeRobot trajectory (default:
episode 0, near steps 14-16). Quantifies differences between consecutive
8D TCP/gripper observations and both RGB camera images; optionally writes
a side-by-side image grid for human inspection.

No model/GPU, simulator rollout, writes to the dataset, or retraining.
An abrupt change in expert gripper commands is not by itself an off-by-one
alignment bug: the scripted expert changes phases when its pose tolerance
is met, which may be hard to infer from a single observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rlinf.data.datasets.openpi.v3_stackcube_dataset import (
    V3StackCubeDataset,
    is_lerobot_v3_dataset,
)
from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
from scripts.check_stackcube_pi05_clean_transforms import (
    as_numpy,
    camera_to_hwc_uint8,
)


def load_episode_window(repo_id: str, episode_id: int) -> tuple[int, list[dict]]:
    """Return the first close timestep and a few untouched stored frames."""
    root = resolve_lerobot_dataset_root(repo_id)
    if is_lerobot_v3_dataset(root):
        dataset = V3StackCubeDataset(
            root, action_horizon=10, use_wrist_image=True,
            state_dim=8, action_dim=7,
        )
        indices = [
            i for i, row in enumerate(dataset.frames)
            if int(row["episode_index"]) == episode_id
        ]
        actions = np.asarray(
            [dataset.frames[i]["actions"] for i in indices], dtype=np.float32,
        )
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=repo_id, root=root)
        flat = dataset.hf_dataset
        indices = np.flatnonzero(
            np.asarray(flat["episode_index"], dtype=np.int64) == episode_id,
        ).tolist()
        actions = np.asarray(
            [flat[int(i)]["actions"] for i in indices], dtype=np.float32,
        )

    if not indices or actions.shape != (len(indices), 7):
        raise ValueError(
            f"Episode {episode_id} missing or has invalid 7D actions: {actions.shape}"
        )
    closes = np.flatnonzero(actions[:, 6] < -0.5)
    if len(closes) == 0:
        raise ValueError(f"Episode {episode_id} has no gripper-close command")
    first_close = int(closes[0])
    if first_close < 1 or actions[first_close - 1, 6] <= 0.5:
        raise ValueError("Expected a +1 (open) to -1 (closed) expert transition")

    selected = range(max(0, first_close - 3), min(len(indices), first_close + 3))
    frames = []
    for step in selected:
        saved = dataset[int(indices[step])]
        state = as_numpy(saved["state"]).astype(np.float64).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"Invalid stored 8D EE state at step {step}: {state.shape}")
        cameras = {
            "base": camera_to_hwc_uint8(saved["image"]),
            "wrist": camera_to_hwc_uint8(saved["wrist_image"]),
        }
        frames.append({
            "step": step, "state": state,
            "action": actions[step].astype(np.float64),
            "cameras": cameras,
        })
    return first_close, frames


def orientation_change_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Shortest orientation distance between WXYZ quaternions."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    n1, n2 = np.linalg.norm(q1), np.linalg.norm(q2)
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Stored quaternion is zero")
    cosine_half_angle = np.clip(abs(float(q1 @ q2 / (n1 * n2))), 0, 1)
    return float(np.degrees(2 * np.arccos(cosine_half_angle)))


def image_change(first: np.ndarray, second: np.ndarray) -> dict:
    """Lossless numeric differences in decoded 224x224 uint8 RGB frames."""
    if first.shape != (224, 224, 3) or second.shape != first.shape:
        raise ValueError("Two comparable 224x224 RGB camera frames are required")
    delta = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return {
        "exact_same_image": bool(np.array_equal(first, second)),
        "mae_uint8": float(delta.mean()),
        "max_abs_uint8": int(delta.max()),
        "fraction_rgb_pixels_changed_gt8": float(
            np.mean(np.any(delta > 8, axis=-1))
        ),
    }


def summarize(first_close: int, frames: list[dict]) -> dict:
    """Report evidence without inventing a closeness/observability threshold."""
    if len(frames) < 2:
        raise ValueError("Need at least two frames to inspect phase transition")
    rows, pairs = [], []
    for frame in frames:
        state = frame["state"]
        action = frame["action"]
        rows.append({
            "step": frame["step"],
            "tcp_xyz_m": state[:3].tolist(),
            "tcp_quat_wxyz": state[3:7].tolist(),
            "gripper_opening_width_mm": float(1000 * state[7]),
            "expert_ee_delta_xyz": action[:3].tolist(),
            "expert_ee_delta_euler": action[3:6].tolist(),
            "expert_gripper_command": float(action[6]),
        })
        print(
            f"FRAME step={frame['step']:2d} "
            f"tcp_xyz={np.round(state[:3], 5).tolist()} "
            f"width_mm={1000 * state[7]:.3f} "
            f"expert_xyz={np.round(action[:3], 3).tolist()} "
            f"expert_grip={action[6]:+.1f}"
        )

    for previous, current in zip(frames[:-1], frames[1:]):
        a, b = previous["state"], current["state"]
        camera_delta = {
            key: image_change(previous["cameras"][key], current["cameras"][key])
            for key in ("base", "wrist")
        }
        row = {
            "from_step": previous["step"], "to_step": current["step"],
            "expert_gripper_before": float(previous["action"][6]),
            "expert_gripper_after": float(current["action"][6]),
            "tcp_translation_mm": float(1000 * np.linalg.norm(b[:3] - a[:3])),
            "tcp_rotation_deg": orientation_change_deg(a[3:7], b[3:7]),
            "gripper_width_change_mm": float(1000 * (b[7] - a[7])),
            "state_exactly_equal": bool(np.array_equal(a, b)),
            "cameras": camera_delta,
        }
        row["full_observation_exactly_equal"] = (
            row["state_exactly_equal"]
            and all(data["exact_same_image"] for data in camera_delta.values())
        )
        pairs.append(row)
        print(
            f"PAIR {previous['step']:2d}->{current['step']:2d} "
            f"grip={row['expert_gripper_before']:+.1f}->{row['expert_gripper_after']:+.1f} "
            f"tcp_mm={row['tcp_translation_mm']:.4f} "
            f"rot_deg={row['tcp_rotation_deg']:.4f} "
            f"width_delta_mm={row['gripper_width_change_mm']:+.4f} "
            f"base_mae={camera_delta['base']['mae_uint8']:.3f} "
            f"wrist_mae={camera_delta['wrist']['mae_uint8']:.3f} "
            f"base_changed_gt8={camera_delta['base']['fraction_rgb_pixels_changed_gt8']:.4f} "
            f"wrist_changed_gt8={camera_delta['wrist']['fraction_rgb_pixels_changed_gt8']:.4f} "
            f"identical={row['full_observation_exactly_equal']}"
        )

    target_pair = next(
        (p for p in pairs if p["to_step"] == first_close), None
    )
    if target_pair is None:
        raise AssertionError("Window omitted the crucial pre-close/close pair")
    print(
        f"FIRST_CLOSE step={first_close} "
        f"prev_and_close_observations_identical={target_pair['full_observation_exactly_equal']}"
    )
    return {
        "first_close_step": first_close,
        "frames": rows,
        "consecutive_pairs": pairs,
        "preclose_to_close": target_pair,
    }


def save_image_grid(frames: list[dict], path: Path) -> None:
    """Arrange the *saved*, unaltered RGB camera views for visual inspection."""
    from PIL import Image, ImageDraw

    width, height = 448, len(frames) * 248
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate(frames):
        y = i * 248
        draw.text((5, y + 4), f"step {frame['step']} base   expert grip {frame['action'][6]:+.1f}", fill="black")
        draw.text((230, y + 4), "wrist", fill="black")
        for j, key in enumerate(("base", "wrist")):
            canvas.paste(Image.fromarray(frame["cameras"][key]), (j * 224, y + 24))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/stackcube_pi05_clean_v1")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    parser.add_argument("--image-grid", type=Path, help="Optional side-by-side PNG")
    args = parser.parse_args()
    if args.episode < 0:
        raise ValueError("--episode must be nonnegative")
    first_close, frames = load_episode_window(args.repo_id, args.episode)
    result = summarize(first_close, frames)
    result["dataset"] = args.repo_id
    result["episode"] = args.episode

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.image_grid is not None:
        save_image_grid(frames, args.image_grid.expanduser().resolve())
    print(f"TRANSITION OBSERVABILITY CHECK COMPLETE episode={args.episode}")


if __name__ == "__main__":
    main()
