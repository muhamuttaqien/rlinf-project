# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Export recorded ManiSkill StackCube observations/actions for world-model research.

This exporter does not train or fine-tune Wan/OpenSora. It expects trajectories
recorded with RGB observations (not state-only trajectories).
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _rgb_dataset(group, camera):
    """Find an RGB array in either standard ManiSkill observation layout."""
    candidates = (
        f"obs/sensor_data/{camera}/rgb",
        f"obs/{camera}/rgb",
    )
    for key in candidates:
        if key in group:
            return group[key][()], key
    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/rgb"):
            found.append(name)

    group.visititems(visitor)
    raise ValueError(
        f"No RGB observation for camera {camera!r}. Found RGB paths: {found}. "
        "Record trajectories with obs_mode='rgb' or replay state trajectories "
        "using ManiSkill's trajectory replay with RGB observations enabled."
    )


def _strip_batch(array, kind):
    array = np.asarray(array)
    if array.ndim >= 2 and array.shape[1] == 1:
        # ManiSkill recordings can retain a singleton vectorized-env axis.
        # Only strip it where the expected rank makes that axis unambiguous.
        if (kind == "rgb" and array.ndim == 5) or (
            kind == "actions" and array.ndim == 3
        ):
            array = array[:, 0]
    return array


def export_episode(group, camera):
    """Return aligned RGB frames and continuous actions from one episode."""
    if "actions" not in group:
        raise ValueError(f"Episode {group.name} does not contain actions")
    rgb, source = _rgb_dataset(group, camera)
    frames = _strip_batch(rgb, "rgb")
    actions = _strip_batch(group["actions"][()], "actions")
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ValueError(f"{source}: expected [T,H,W,3/4], got {frames.shape}")
    if actions.ndim != 2:
        raise ValueError(f"{group.name}/actions: expected [T,A], got {actions.shape}")
    if actions.shape[1] != 7:
        raise ValueError(
            f"{group.name}/actions: expected 7-D Panda EE action, got {actions.shape[1]}. "
            "Verify control_mode='pd_ee_delta_pose'."
        )
    if len(frames) != len(actions) + 1:
        raise ValueError(
            f"{group.name}: expected T+1 RGB observations for T actions, "
            f"got {len(frames)} frames and {len(actions)} actions."
        )
    if frames.dtype != np.uint8:
        raise ValueError(f"{source}: expected uint8 RGB images, got {frames.dtype}")
    return frames[..., :3], actions.astype(np.float32)


def convert(input_path, output_dir, camera="base_camera"):
    """Export all recorded episodes and write a machine-readable manifest."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(input_path), "camera": camera, "episodes": []}
    with h5py.File(input_path, "r") as file:
        names = sorted(
            (name for name in file if isinstance(file[name], h5py.Group) and name.startswith("traj_")),
            key=lambda name: int(name[5:]) if name[5:].isdigit() else name,
        )
        if not names:
            raise ValueError("No traj_* groups found in the ManiSkill HDF5 recording")
        for name in names:
            frames, actions = export_episode(file[name], camera)
            path = output_dir / f"{name}.npz"
            np.savez_compressed(path, rgb=frames, actions=actions)
            manifest["episodes"].append(
                {"episode": name, "path": path.name, "steps": len(actions),
                 "action_dim": actions.shape[1], "rgb_shape": list(frames.shape)}
            )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="ManiSkill .h5 recording")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera", default="base_camera")
    args = parser.parse_args()
    manifest = convert(args.input, args.output, args.camera)
    print(f"Exported {len(manifest['episodes'])} episodes to {args.output}")


if __name__ == "__main__":
    main()
