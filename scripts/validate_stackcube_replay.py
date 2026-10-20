# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Inspect replayed ManiSkill trajectories before converting to LeRobot."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def walk(group, prefix=""):
    """Yield named HDF5 datasets, recursively."""
    for name, node in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(node, h5py.Dataset):
            yield path, node
        elif isinstance(node, h5py.Group):
            yield from walk(node, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(
            "outputs/stackcube_recap/raw/StackCube-v1/motionplanning/"
            "stackcube_expert.rgb.pd_ee_delta_pose.physx_cpu.h5"
        ),
    )
    parser.add_argument("--expected-episodes", type=int, default=10)
    parser.add_argument("--expected-actions", type=int, default=7)
    args = parser.parse_args()
    failures = []
    with h5py.File(args.path, "r") as handle:
        episodes = sorted(
            (k for k in handle if k.startswith("traj_")),
            key=lambda k: int(k.split("_")[-1]),
        )
        print(f"File: {args.path}\nEpisodes: {len(episodes)}")
        if len(episodes) != args.expected_episodes:
            failures.append(
                f"Expected {args.expected_episodes} episodes, got {len(episodes)}"
            )
        summary = []
        for name in episodes:
            group = handle[name]
            datasets = dict(walk(group))
            if "actions" not in datasets:
                failures.append(f"{name}: missing actions")
                continue
            actions = datasets["actions"]
            n_steps = actions.shape[0]
            action_dim = actions.shape[-1]
            if action_dim != args.expected_actions:
                failures.append(
                    f"{name}: action dimension {action_dim}, "
                    f"expected {args.expected_actions}"
                )
            obs_paths = [k for k in datasets if k.startswith("obs/")]
            image_paths = [
                k for k in obs_paths
                if ("rgb" in k.lower() or "image" in k.lower())
                and datasets[k].ndim >= 4
                and datasets[k].shape[-1] in (3, 4)
            ]
            if not obs_paths:
                failures.append(f"{name}: missing observation arrays")
            if not image_paths:
                failures.append(f"{name}: no RGB image tensors found")
            for k in obs_paths:
                length = datasets[k].shape[0] if datasets[k].ndim else 0
                if length not in (n_steps, n_steps + 1):
                    failures.append(
                        f"{name}: {k} length {length} does not align with "
                        f"{n_steps} actions"
                    )
            if "success" not in datasets:
                failures.append(f"{name}: missing success flag")
                terminal_success = False
            else:
                success = np.asarray(datasets["success"][()]).reshape(-1)
                terminal_success = bool(success[-1]) if success.size else False
                if not terminal_success:
                    failures.append(f"{name}: final success flag is false")
            summary.append(
                {
                    "episode": name,
                    "steps": int(n_steps),
                    "action_dim": int(action_dim),
                    "terminal_success": terminal_success,
                    "rgb_paths": image_paths,
                }
            )
            print(
                f"{name}: steps={n_steps} action_dim={action_dim} "
                f"terminal_success={terminal_success} "
                f"rgb_streams={len(image_paths)}"
            )
            if name == episodes[0]:
                print("First episode dataset schema:")
                for path, dataset in datasets.items():
                    print(f"  {path}: shape={dataset.shape}, dtype={dataset.dtype}")
    print(json.dumps({"episodes": summary, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(f"Validation FAILED: {len(failures)} issue(s)")
    print("Validation PASSED")


if __name__ == "__main__":
    main()
