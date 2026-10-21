# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Validate locally converted Panda StackCube LeRobot episodes and video decoding."""

import argparse
from pathlib import Path

import h5py
import numpy as np


CAMERAS = ("observation.images.base_camera", "observation.images.hand_camera")


def array(value):
    """Turn torch or numpy frame fields into numpy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def assert_shape(value, shape, label):
    """Fail with a useful message if a feature shape changed."""
    if tuple(array(value).shape) != tuple(shape):
        raise ValueError(f"{label}: expected shape {shape}, got {array(value).shape}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/stackcube_recap/lerobot_stackcube_v2"),
    )
    parser.add_argument("--repo-id", default="local/stackcube-panda-recap")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "outputs/stackcube_recap/raw/StackCube-v1/motionplanning/"
            "stackcube_expert.rgb.pd_ee_delta_pose.physx_cpu.h5"
        ),
    )
    parser.add_argument("--expected-episodes", type=int, default=10)
    parser.add_argument("--expected-fps", type=int, default=20)
    parser.add_argument(
        "--video-backend",
        choices=("torchcodec", "pyav"),
        default=None,
        help="Choose a LeRobot decoder (default: its automatic selection)",
    )
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not (args.root / "meta" / "info.json").is_file():
        parser.error(f"No LeRobot metadata at {args.root}")
    if not args.source.is_file():
        parser.error(f"Source HDF5 missing: {args.source}")

    # Local root takes precedence; no Hugging Face Hub upload is performed.
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        video_backend=args.video_backend,
    )
    print(f"Dataset root: {args.root}")
    print(f"Episodes: {dataset.num_episodes}, frames: {len(dataset)}, fps: {dataset.fps}")
    print(f"Video decoder: {dataset.video_backend}")
    print(f"Features: {list(dataset.features)}")

    if dataset.num_episodes != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} episodes, got {dataset.num_episodes}")
    if dataset.fps != args.expected_fps:
        raise ValueError(f"Expected {args.expected_fps} fps, got {dataset.fps}")
    for key in (*CAMERAS, "observation.state", "action", "is_success"):
        if key not in dataset.features:
            raise ValueError(f"Missing LeRobot feature: {key}")

    with h5py.File(args.source, "r") as source:
        episode_names = sorted(
            (k for k in source.keys() if k.startswith("traj_")),
            key=lambda k: int(k.split("_")[-1]),
        )[: args.expected_episodes]
        episode_lengths = [len(source[k]["actions"]) for k in episode_names]
        if len(episode_lengths) != args.expected_episodes:
            raise ValueError("Source has fewer episodes than requested")
        if len(dataset) != sum(episode_lengths):
            raise ValueError(
                f"Frame count mismatch: LeRobot {len(dataset)}, "
                f"source {sum(episode_lengths)}"
            )

        offset = 0
        for episode_number, (name, steps) in enumerate(zip(episode_names, episode_lengths)):
            indices = sorted({offset, offset + steps // 2, offset + steps - 1})
            for index in indices:
                # __getitem__ decodes both RGB videos, not only parquet records.
                try:
                    frame = dataset[index]
                except Exception as error:
                    raise RuntimeError(
                        f"Video/frame decoding failed at {name}, global index {index}. "
                        "If using torchcodec, retry with --video-backend pyav."
                    ) from error
                actual_episode = int(array(frame["episode_index"]).item())
                if actual_episode != episode_number:
                    raise ValueError(
                        f"Index {index}: expected episode {episode_number}, "
                        f"found {actual_episode}"
                    )
                assert_shape(frame["observation.state"], (18,), "Panda state")
                assert_shape(frame["action"], (7,), "Panda action")
                for key in CAMERAS:
                    assert_shape(frame[key], (3, 128, 128), key)
                    rgb = array(frame[key])
                    if not np.isfinite(rgb).all():
                        raise ValueError(f"{name}: non-finite {key}")
                    if float(rgb.max()) <= float(rgb.min()):
                        raise ValueError(f"{name}: constant decoded camera image: {key}")
                for key in ("observation.state", "action"):
                    if not np.isfinite(array(frame[key])).all():
                        raise ValueError(f"{name}: non-finite {key}")
                local_index = index - offset
                raw_action = np.asarray(source[name]["actions"][local_index])
                if not np.allclose(array(frame["action"]), raw_action, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"{name}: action mismatch at frame {local_index}")
                if not isinstance(frame["task"], str) or not frame["task"].strip():
                    raise ValueError(f"{name}: missing task instruction")

            last = dataset[offset + steps - 1]
            if not bool(array(last["is_success"]).reshape(-1)[-1]):
                raise ValueError(f"{name}: terminal success flag is false")
            print(f"{name}: {steps} frames, cameras decode, terminal success=True")
            offset += steps

    print(f"Validation PASSED: {dataset.num_episodes} episodes, {len(dataset)} frames")


if __name__ == "__main__":
    main()
