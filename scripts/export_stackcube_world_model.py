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
from PIL import Image
from tqdm.auto import tqdm


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


def export_episode(group, camera, image_size=256):
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
    frames = frames[..., :3]
    if frames.shape[1:3] != (image_size, image_size):
        frames = np.stack([
            np.asarray(Image.fromarray(frame).resize((image_size, image_size), Image.Resampling.BILINEAR))
            for frame in frames
        ])
    return frames, actions.astype(np.float32)


def convert(input_path, output_dir, camera="base_camera", image_size=256, cameras=None):
    """Export all recorded episodes and write a machine-readable manifest."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_names = tuple(cameras) if cameras is not None else (camera,)
    if not camera_names or len(set(camera_names)) != len(camera_names):
        raise ValueError("Specify at least one camera and do not repeat camera names")
    if any(name not in ("base_camera", "hand_camera") for name in camera_names):
        raise ValueError("Supported cameras are base_camera and hand_camera")
    multi_view = cameras is not None
    manifest = {"source": str(input_path), "cameras": list(camera_names), "image_size": image_size, "episodes": []}
    with h5py.File(input_path, "r") as file:
        names = sorted(
            (name for name in file if isinstance(file[name], h5py.Group) and name.startswith("traj_")),
            key=lambda name: int(name[5:]) if name[5:].isdigit() else name,
        )
        if not names:
            raise ValueError("No traj_* groups found in the ManiSkill HDF5 recording")
        for name in tqdm(names, desc="Exporting StackCube", unit="episode", dynamic_ncols=True):
            episode = file[name]
            arrays = {}
            shapes = {}
            actions = None
            for camera_name in camera_names:
                frames, camera_actions = export_episode(episode, camera_name, image_size)
                if actions is None:
                    actions = camera_actions
                elif not np.array_equal(actions, camera_actions):
                    raise ValueError(f"{name}: action arrays mismatch between cameras")
                key = "rgb" if not multi_view else (
                    "base_rgb" if camera_name == "base_camera" else "wrist_rgb"
                )
                arrays[key] = frames
                shapes[key] = list(frames.shape)
            arrays["actions"] = actions
            path = output_dir / f"{name}.npz"
            # Compression can be time-consuming for large RGB datasets;
            # tqdm updates after each episode has been fully written.
            np.savez_compressed(path, **arrays)
            manifest["episodes"].append(
                {"episode": name, "path": path.name, "steps": len(actions),
                 "action_dim": actions.shape[1], "rgb_shapes": shapes}
            )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="ManiSkill .h5 recording")
    parser.add_argument("--output", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--camera", choices=("base_camera", "hand_camera"), help="Legacy single-camera export")
    selection.add_argument("--cameras", nargs="+", choices=("base_camera", "hand_camera"), help="Cameras to export (default: both)")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    cameras = args.cameras if args.cameras is not None else (
        None if args.camera is not None else ["base_camera", "hand_camera"]
    )
    manifest = convert(args.input, args.output, camera=args.camera or "base_camera",
                       image_size=args.image_size, cameras=cameras)
    print(f"Exported {len(manifest['episodes'])} episodes to {args.output}")


if __name__ == "__main__":
    main()
