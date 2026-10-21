# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Compute OpenPI-compatible state/action norm_stats.json from StackCube HDF5.

The stackcube_panda adapter makes no additional delta/rotation transforms:
qpos+qvel -> state and pd_ee_delta_pose -> action. Statistics therefore
can be computed directly from the validated replay, before image decoding.
Actions are weighted as OpenPI's H-step, end-padded action windows.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from openpi.shared import normalize


def unbatch(data, steps):
    """Strip a leading singleton parallel-env dimension from ManiSkill frames."""
    data = np.asarray(data)
    if data.shape[0] not in (steps, steps + 1):
        raise ValueError(f"Unexpected trajectory length {data.shape[0]} != {steps}")
    if data.ndim >= 3 and data.shape[1] == 1:
        data = data[:, 0]
    return data[:steps]


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
        default=Path("outputs/stackcube_recap/openpi_norm/stackcube_panda"),
    )
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if (args.output / "norm_stats.json").exists():
        parser.error(f"Refusing to overwrite {args.output / 'norm_stats.json'}")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    total_frames = 0
    with h5py.File(args.input, "r") as h5:
        names = sorted(
            (k for k in h5 if k.startswith("traj_")),
            key=lambda k: int(k.split("_")[-1]),
        )
        for name in names:
            group = h5[name]
            n = len(group["actions"])
            success = np.asarray(group["success"]).reshape(-1)
            if not n or len(success) != n or not bool(success[-1]):
                raise ValueError(f"{name}: missing successful terminal state")
            actions = unbatch(group["actions"][()], n).astype(np.float32)
            if actions.shape != (n, 7):
                raise ValueError(f"{name}: expected ({n}, 7) actions, got {actions.shape}")
            state = np.concatenate(
                [
                    unbatch(group[f"obs/agent/{key}"][()], n)
                    for key in ("qpos", "qvel")
                ],
                axis=-1,
            ).astype(np.float32)
            if state.shape != (n, 18):
                raise ValueError(f"{name}: expected ({n},18) state, got {state.shape}")
            if not np.isfinite(state).all() or not np.isfinite(actions).all():
                raise ValueError(f"{name}: non-finite state/actions")
            # OpenPI creates a future action chunk at every timestep and
            # clamps horizon-overflow indices to the episode's final frame.
            indices = np.minimum(
                np.arange(n)[:, None] + np.arange(args.horizon)[None, :],
                n - 1,
            )
            stats["state"].update(state)
            stats["actions"].update(actions[indices])
            total_frames += n
            print(f"{name}: {n} frames")

    norm_stats = {key: stat.get_statistics() for key, stat in stats.items()}
    normalize.save(args.output, norm_stats)
    print(f"Saved OpenPI normalization stats: {args.output / 'norm_stats.json'}")
    print(f"Frames: {total_frames}; action horizon: {args.horizon}")
    print("Recompute these stats whenever training dataset composition changes.")


if __name__ == "__main__":
    main()
