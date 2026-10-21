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

"""Upload small, real StackCube demonstrations to the training W&B run.

These are dataset examples (expert actions), NOT rollouts from the learned
policy.  This is called on the driver once at run initialization; Ray workers
never import or initialize W&B for media logging.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rlinf.data.storage.lerobot import (
    resolve_lerobot_dataset_root,
    resolve_lerobot_repo_id,
)

logger = logging.getLogger(__name__)


def to_rgb_uint8(image: Any) -> np.ndarray:
    """Convert a LeRobot image tensor, PIL image or numpy image to HWC RGB."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "convert"):
        image = image.convert("RGB")

    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D RGB image, got {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB channels, got {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        # LeRobot image tensors are typically CHW float32 in [0, 1].
        arr = np.clip(arr * 255.0 if np.nanmax(arr) <= 1.0 else arr, 0, 255)
    else:
        arr = np.clip(arr, 0, 255)
    return np.ascontiguousarray(arr.astype(np.uint8))


def evenly_spaced_indices(indices: np.ndarray, limit: int) -> list[int]:
    """Include the beginning and end of an episode while bounding decoding."""
    if limit < 2:
        raise ValueError("frames_per_video must be at least 2")
    count = min(len(indices), limit)
    if count == 0:
        return []
    positions = np.linspace(0, len(indices) - 1, count, dtype=int)
    return [int(indices[p]) for p in positions]


def _get_episode_index(dataset: Any) -> np.ndarray:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise ValueError("LeRobot dataset has no hf_dataset frame metadata")
    if "episode_index" not in hf_dataset.column_names:
        raise ValueError("LeRobot dataset lacks episode_index metadata")
    return np.asarray(hf_dataset["episode_index"], dtype=np.int64).reshape(-1)


def build_stackcube_wandb_media(
    repo_id: str,
    wandb: Any,
    *,
    max_episodes: int = 2,
    frames_per_video: int = 32,
) -> dict[str, Any]:
    """Build snapshot images and short MP4s from *actual* StackCube demonstrations.

    Uses the already-collected LeRobot data; no simulator or trained checkpoint
    is needed.  Only the base image used by the OpenPI policy is visualized.
    """
    if max_episodes <= 0:
        return {}
    resolved = resolve_lerobot_repo_id(repo_id)
    if not resolved:
        raise ValueError("No StackCube dataset path was configured")
    root = resolve_lerobot_dataset_root(resolved)
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset metadata not found: {root}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(resolved, root=root)
    episode_ids = _get_episode_index(dataset)
    available = np.unique(episode_ids)[:max_episodes]
    result: dict[str, Any] = {}
    original_fps = float(getattr(dataset, "fps", 20) or 20)

    for episode_id in available:
        frame_ids = np.flatnonzero(episode_ids == episode_id)
        chosen = evenly_spaced_indices(frame_ids, frames_per_video)
        if not chosen:
            continue

        # This is a bounded decode of 32 frames per episode, not the full dataset.
        frames = [to_rgb_uint8(dataset[frame_id]["image"]) for frame_id in chosen]
        label = f"dataset/expert_episode_{int(episode_id):03d}"
        snapshots = [frames[0], frames[len(frames) // 2], frames[-1]]
        result[f"{label}/snapshots"] = [
            wandb.Image(rgb, caption=f"Expert episode {int(episode_id)}: {stage} (base camera)")
            for stage, rgb in zip(("start", "middle", "end"), snapshots)
        ]

        # Log the entire sampled trajectory at an adjusted playback rate,
        # rather than playing 32 frames as though they were contiguous.
        stride = max(1, int(round(len(frame_ids) / len(chosen))))
        fps = max(1, int(round(original_fps / stride)))
        video = np.stack(frames).transpose(0, 3, 1, 2)  # W&B: T, C, H, W
        try:
            result[f"{label}/demonstration_video"] = wandb.Video(
                video,
                fps=fps,
                format="mp4",
                caption=(
                    f"StackCube successful expert demonstration "
                    f"{int(episode_id)} (sampled)"
                ),
            )
        except Exception as exc:
            logger.warning(
                "Could not encode demonstration MP4 for episode %s (%s); "
                "trying GIF instead.",
                episode_id,
                exc,
            )
            try:
                result[f"{label}/demonstration_video"] = wandb.Video(
                    video,
                    fps=fps,
                    format="gif",
                    caption=(
                        f"StackCube successful expert demonstration "
                        f"{int(episode_id)} (sampled)"
                    ),
                )
            except Exception as fallback_exc:
                logger.warning(
                    "Could not encode demonstration GIF for episode %s: %s",
                    episode_id,
                    fallback_exc,
                )

    return result
