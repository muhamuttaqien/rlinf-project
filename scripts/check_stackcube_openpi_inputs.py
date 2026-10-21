# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Check LeRobot -> OpenPI pi0.5 StackCube preprocessing (no checkpoint/GPU)."""

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def to_numpy(x):
    """Copy a CPU tensor or NumPy object into an array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/stackcube_recap/lerobot_stackcube_v2"),
    )
    parser.add_argument("--repo-id", default="local/stackcube-panda-recap")
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    if not (args.root / "meta" / "info.json").is_file():
        parser.error(f"Dataset not found at {args.root}")

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    if len(ds) == 0:
        raise ValueError("Dataset is empty")
    train_cfg = get_openpi_config(
        "pi05_stackcube_panda",
        repo_id=str(args.root.resolve()),
    )
    if train_cfg.model.action_horizon != args.horizon:
        raise ValueError(
            f"Window {args.horizon} != OpenPI action horizon "
            f"{train_cfg.model.action_horizon}"
        )
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

    # Build an action window from episode 0. Do not cross episode boundaries.
    sample = dict(ds[0])
    end = min(args.horizon, len(ds))
    actions = [to_numpy(ds.hf_dataset[i]["action"]) for i in range(end)]
    while len(actions) < args.horizon:
        actions.append(actions[-1])
    sample["action"] = np.stack(actions).astype(np.float32)

    for transform in data_cfg.repack_transforms.inputs:
        sample = transform(sample)
    for transform in data_cfg.data_transforms.inputs:
        sample = transform(sample)

    state = to_numpy(sample["state"])
    actions = to_numpy(sample["actions"])
    expected_image_keys = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    print("Prompt:", sample.get("prompt"))
    print("State:", state.shape)
    print("Actions:", actions.shape)
    for key in expected_image_keys:
        image = to_numpy(sample["image"][key])
        print(f"{key}: shape={image.shape}, dtype={image.dtype}, "
              f"enabled={bool(sample['image_mask'][key])}")
        if image.shape != (128, 128, 3) or image.dtype != np.uint8:
            raise ValueError(f"Unexpected {key} image format")
    if state.shape != (18,) or actions.shape != (args.horizon, 7):
        raise ValueError("Bad state/action dimensions")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite state or actions")
    if not sample.get("prompt"):
        raise ValueError("Language instruction missing")
    if not bool(sample["image_mask"]["left_wrist_0_rgb"]):
        raise ValueError("Wrist image was not activated")
    print("OpenPI StackCube input adapter PASSED (pre-model-transform check)")
    print("No checkpoint was loaded; tokenization, norm stats, and GPU training remain untested.")


if __name__ == "__main__":
    main()
