# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Convert validated StackCube RGB HDF5 into a local LeRobot dataset."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def read_frame(dataset, i, steps):
    """Remove ManiSkill's leading singleton environment dimension."""
    if dataset.shape[0] not in (steps, steps + 1):
        raise ValueError(f"Misaligned array shape {dataset.shape}, steps={steps}")
    item = np.asarray(dataset[i])
    if item.ndim >= 1 and item.shape[0] == 1:
        item = item[0]
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/stackcube_recap/raw/StackCube-v1/motionplanning/"
            "stackcube_expert.rgb.pd_ee_delta_pose.physx_cpu.h5"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stackcube_recap/lerobot_stackcube_v1"),
    )
    parser.add_argument("--repo-id", default="local/stackcube-panda-recap")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--task", default="Stack the red cube on the green cube")
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"Output exists: {args.output}. Choose a fresh --output.")
    if args.fps < 1:
        parser.error("--fps must be positive")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise SystemExit(
            f"Unable to import LeRobotDataset: {type(error).__name__}: {error}. "
            "LeRobot may be installed, but its dataset dependencies or API may differ."
        ) from error

    camera_paths = {
        "observation.images.base_camera": "obs/sensor_data/base_camera/rgb",
        "observation.images.hand_camera": "obs/sensor_data/hand_camera/rgb",
    }
    with h5py.File(args.input, "r") as h5:
        names = sorted(
            [k for k in h5 if k.startswith("traj_")],
            key=lambda k: int(k.split("_")[-1]),
        )
        if args.max_episodes:
            names = names[: args.max_episodes]
        if not names:
            raise ValueError("No trajectories found")
        first = h5[names[0]]
        steps = len(first["actions"])
        # Only robot proprioception: do not leak object poses into the policy.
        state_keys = [
            k for k in ("qpos", "qvel") if f"obs/agent/{k}" in first
        ]
        if not state_keys:
            raise ValueError(
                "No obs/agent/qpos or obs/agent/qvel found. Inspect the schema "
                "and explicitly map Panda proprioception before proceeding."
            )
        def state_at(g, i, n):
            return np.concatenate(
                [
                    read_frame(g[f"obs/agent/{k}"], i, n).reshape(-1)
                    for k in state_keys
                ]
            ).astype(np.float32)
        state_dim = len(state_at(first, 0, steps))
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"state_{i}" for i in range(state_dim)],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"action_{i}" for i in range(7)],
            },
            "is_success": {"dtype": "bool", "shape": (1,), "names": None},
        }
        for key, path in camera_paths.items():
            if path not in first:
                raise ValueError(f"Missing camera: {path}")
            img = read_frame(first[path], 0, steps)
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(f"{path} shape {img.shape} is not HWC RGB")
            features[key] = {
                "dtype": "video",
                "shape": img.shape,
                "names": ["height", "width", "channel"],
            }
        print(f"Robot state keys: {state_keys}, flattened dimension: {state_dim}")
        print(f"Cameras: {[features[k]['shape'] for k in camera_paths]}")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.output,
            fps=args.fps,
            robot_type="panda_wristcam",
            features=features,
            use_videos=True,
        )
        try:
            for name in names:
                g = h5[name]
                n = len(g["actions"])
                success = np.asarray(g["success"]).reshape(-1)
                if len(success) != n or not bool(success[-1]):
                    raise ValueError(f"{name}: invalid terminal success")
                if g["actions"].shape[-1] != 7:
                    raise ValueError(f"{name}: expected seven action dimensions")
                for i in range(n):
                    frame = {
                        "observation.state": state_at(g, i, n),
                        "action": np.asarray(g["actions"][i], dtype=np.float32),
                        "is_success": np.asarray([bool(success[i])], dtype=np.bool_),
                        "task": args.task,
                    }
                    for key, path in camera_paths.items():
                        frame[key] = read_frame(g[path], i, n).astype(np.uint8)
                    dataset.add_frame(frame)
                dataset.save_episode()
                print(f"Saved {name}: {n} frames")
        finally:
            if hasattr(dataset, "finalize"):
                dataset.finalize()
        print(f"LeRobot dataset saved: {args.output}")
        print(
            "NOTICE: Verify episode count, video decode, and observation/action "
            "normalization before configuring RLinf OpenPI SFT."
        )


if __name__ == "__main__":
    main()
