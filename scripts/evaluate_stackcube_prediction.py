# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Evaluate a transparent persistence baseline on indexed StackCube windows.

This is a reference metric and data-pipeline verification, NOT Wan/OpenSora
inference. A world model must beat this no-motion baseline on held-out data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def evaluate(index_path, output, max_windows=50, seed=0):
    """Compute per-view persistence MAE using deterministic sampled windows."""
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    windows = index["windows"]
    if not windows:
        raise ValueError("Window index is empty")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(windows), size=min(len(windows), max_windows), replace=False)
    source = Path(index["source_dir"])
    cond = int(index["condition_frames"])
    chunk = int(index["chunk"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    totals = {"base": 0., "wrist": 0.}
    count = 0
    first = None
    for n in tqdm(chosen, desc="Evaluating persistence", unit="window"):
        item = windows[int(n)]
        t = int(item["action_start"])
        with np.load(source / item["file"], allow_pickle=False) as data:
            if data["actions"][t:t+chunk].shape != (chunk, 7):
                raise ValueError(f"Missing action chunk in {item['file']} at {t}")
            for label, key in (("base", "base_rgb"), ("wrist", "wrist_rgb")):
                frames = data[key]
                initial = frames[t]
                targets = frames[t+1:t+chunk+1]
                if targets.shape != (chunk, 256, 256, 3):
                    raise ValueError(f"Unexpected targets {targets.shape} in {item['file']}")
                # float32 avoids unsigned integer wraparound in differences.
                totals[label] += float(
                    np.abs(targets.astype(np.float32) - initial.astype(np.float32))
                    .mean(axis=(1, 2, 3)).sum()
                )
                if first is None and label == "base":
                    first = {"condition": frames[t-cond+1:t+1].copy(),
                             "base_initial": initial.copy(), "base_target": targets[-1].copy()}
                if first is not None and "wrist_initial" not in first and label == "wrist":
                    first["wrist_initial"] = initial.copy()
                    first["wrist_target"] = targets[-1].copy()
        count += chunk
    if first is None or "wrist_initial" not in first:
        raise RuntimeError("Failed to gather visualization frames")
    # Side-by-side last observed, final actual, and unchanged persistence prediction.
    for label in ("base", "wrist"):
        panel = np.concatenate(
            [first[f"{label}_initial"], first[f"{label}_target"],
             first[f"{label}_initial"]], axis=1
        )
        Image.fromarray(panel).save(output / f"{label}_persistence_comparison.png")
    result = {
        "evaluation": "persistence_baseline_not_world_model",
        "index": str(index_path),
        "num_windows": len(chosen),
        "action_chunk": chunk,
        "base_mae_0_255": totals["base"] / count,
        "wrist_mae_0_255": totals["wrist"] / count,
        "notes": "MAE compares all 8 future real frames with repeated latest input frame."
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = evaluate(args.index, args.output, args.max_windows, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
