# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Prepare RLinf world-model reset frames and OpenSora action statistics.

Consumes validated dual-view StackCube .npz episodes. Reset .npy files use
RLinf's NpyTrajectoryDatasetWrapper schema; they are not Wan/OpenSora
training sets or fine-tuned checkpoints.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def prepare(source, destination, task="Stack the red cube on the green cube."):
    """Build lightweight reset trajectories and q01/q99 action normalization."""
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "manifest.json").read_text())
    episodes = manifest["episodes"]
    if not episodes:
        raise ValueError("Input manifest contains no episodes")
    resets = destination / "initial_frames"
    resets.mkdir(parents=True, exist_ok=True)
    action_arrays = []
    records = []
    for item in tqdm(episodes, desc="Preparing WM inputs", unit="episode"):
        with np.load(source / item["path"], allow_pickle=False) as episode:
            keys = set(episode.files)
            if not {"base_rgb", "wrist_rgb", "actions"} <= keys:
                raise ValueError(f"{item['path']}: dual-view keys are required")
            base = episode["base_rgb"]
            wrist = episode["wrist_rgb"]
            actions = episode["actions"]
            if (base.ndim != 4 or wrist.shape != base.shape or
                    base.shape[1:] != (256, 256, 3) or
                    actions.shape != (len(base) - 1, 7)):
                raise ValueError(f"{item['path']}: invalid RGB/action alignment")
            if base.dtype != np.uint8 or wrist.dtype != np.uint8:
                raise ValueError(f"{item['path']}: RGB frames must be uint8")
            if not np.isfinite(actions).all():
                raise ValueError(f"{item['path']}: non-finite actions")
            action_arrays.append(actions.astype(np.float32))
            # Existing RLinf reset loader expects at least one frame containing
            # 'image', 'delta_action' and a text task description.
            init = {"image": base[0].copy(), "wrist_rgb": wrist[0].copy(),
                    "delta_action": np.zeros(7, dtype=np.float32),
                    "instruction": task}
            np.save(resets / f"{item['episode']}.npy",
                    np.array([init], dtype=object), allow_pickle=True)
            records.append({"episode": item["episode"], "steps": len(actions),
                            "source": item["path"]})
    actions = np.concatenate(action_arrays)
    q01, q99 = np.quantile(actions, [0.01, 0.99], axis=0)
    # OpenSora currently divides by q99-q01 without epsilon.
    # Reject degenerate dimensions instead of emitting invalid statistics.
    degenerate = np.flatnonzero(q99 - q01 < 1e-8)
    if len(degenerate):
        raise ValueError(
            f"Degenerate action quantiles at dimensions {degenerate.tolist()}; "
            "OpenSora normalization requires a separately reviewed strategy."
        )
    stats = {"action": {"q01": q01.tolist(), "q99": q99.tolist()}}
    (destination / "action_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (destination / "dataset_index.json").write_text(
        json.dumps({"task": task, "cameras": ["base_camera", "hand_camera"],
                    "num_episodes": len(records), "episodes": records}, indent=2) + "\n"
    )
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", default="Stack the red cube on the green cube.")
    args = parser.parse_args()
    count = prepare(args.input, args.output, args.task)
    print(f"Prepared {count} world-model reset episodes in {args.output}")


if __name__ == "__main__":
    main()
