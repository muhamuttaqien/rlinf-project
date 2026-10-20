# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Index aligned StackCube action-conditioned training windows for Wan/OpenSora.

This stage builds window metadata and a lazy loader; it does not fine-tune
video diffusion checkpoints or enable multi-camera inference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

SPECS = {"wan": 5, "opensora": 4}
CHUNK = 8


def build_index(source, output, backend, stride=8):
    """Index [condition frames, eight target frames, eight actions] windows."""
    if backend not in SPECS:
        raise ValueError(f"backend must be one of {tuple(SPECS)}")
    if stride <= 0:
        raise ValueError("stride must be positive")
    source, output = Path(source), Path(output)
    manifest = json.loads((source / "manifest.json").read_text())
    cond = SPECS[backend]
    windows = []
    for episode in tqdm(manifest["episodes"], desc=f"Indexing {backend}", unit="episode"):
        path = source / episode["path"]
        with np.load(path, allow_pickle=False) as data:
            if not {"base_rgb", "wrist_rgb", "actions"} <= set(data.files):
                raise ValueError(f"{path}: missing dual-view data")
            actions = data["actions"]
            base = data["base_rgb"]
            wrist = data["wrist_rgb"]
            if (actions.ndim != 2 or actions.shape[1] != 7 or
                    base.shape != wrist.shape or base.shape[1:] != (256, 256, 3) or
                    len(base) != len(actions) + 1):
                raise ValueError(f"{path}: inconsistent shapes")
            if not np.isfinite(actions).all():
                raise ValueError(f"{path}: non-finite actions")
            # target observations are o[t+1..t+8]; action chunk is a[t..t+7].
            # Require full real context for an offline supervised window.
            for t in range(cond - 1, len(actions) - CHUNK + 1, stride):
                windows.append({"episode": episode["episode"], "file": episode["path"],
                                "action_start": t, "condition_start": t-cond+1,
                                "action_count": CHUNK, "condition_count": cond})
    if not windows:
        raise ValueError("No complete windows: trajectories may be too short")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"backend": backend, "condition_frames": cond, "chunk": CHUNK,
              "stride": stride, "source_dir": str(source.resolve()),
              "num_windows": len(windows), "windows": windows}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def read_window(index_path, window_id):
    """Load one window without materializing every episode in RAM."""
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    item = index["windows"][window_id]
    cond = index["condition_frames"]
    t = item["action_start"]
    with np.load(Path(index["source_dir"]) / item["file"], allow_pickle=False) as data:
        return {"base_condition": data["base_rgb"][t-cond+1:t+1],
                "wrist_condition": data["wrist_rgb"][t-cond+1:t+1],
                "base_target": data["base_rgb"][t+1:t+CHUNK+1],
                "wrist_target": data["wrist_rgb"][t+1:t+CHUNK+1],
                "actions": data["actions"][t:t+CHUNK]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=tuple(SPECS), required=True)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    result = build_index(args.input, args.output, args.backend, args.stride)
    print(f"Indexed {result['num_windows']} {args.backend} windows in {args.output}")


if __name__ == "__main__":
    main()
