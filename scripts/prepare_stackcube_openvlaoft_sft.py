# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Prepare episode-separated StackCube demonstrations for an OpenVLA-OFT SFT baseline.

Intermediate dataset only: training must use an OpenVLA-OFT-compatible
processor, robot action normalization and checkpoint. No policy is trained.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm


def prepare(source, output, *, val_fraction=0.1, seed=42, chunk=8):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    source, output = Path(source), Path(output)
    entries = json.loads((source / "manifest.json").read_text())["episodes"]
    if len(entries) < 2:
        raise ValueError("At least two episodes are needed for train/validation split")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(entries))
    val_ids = set(int(i) for i in permutation[:max(1, min(len(entries)-1, round(len(entries)*val_fraction)))])
    output.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": []}
    all_training_actions = []
    for idx, entry in enumerate(tqdm(entries, desc="Indexing OpenVLA-OFT SFT", unit="episode")):
        path = source / entry["path"]
        with np.load(path, allow_pickle=False) as data:
            if not {"base_rgb", "wrist_rgb", "actions"} <= set(data.files):
                raise ValueError(f"{path}: both cameras and actions are required")
            base, wrist, actions = data["base_rgb"], data["wrist_rgb"], data["actions"]
            if (base.ndim != 4 or base.shape[1:] != (256, 256, 3) or
                    wrist.shape != base.shape or
                    actions.ndim != 2 or actions.shape != (len(base)-1, 7)):
                raise ValueError(f"{path}: invalid action/observation alignment")
            if not np.isfinite(actions).all():
                raise ValueError(f"{path}: actions contain non-finite values")
            split = "val" if idx in val_ids else "train"
            if split == "train":
                all_training_actions.append(actions.astype(np.float32))
            for t in range(len(actions) - chunk + 1):
                splits[split].append({
                    "episode": entry["episode"], "file": entry["path"],
                    "step": t, "action_start": t, "action_count": chunk,
                    "observation_frame": t,
                    "instruction": "Stack the red cube on top of the green cube."
                })
    if not splits["train"] or not splits["val"]:
        raise ValueError("No complete action chunks in training or validation split")
    training_actions = np.concatenate(all_training_actions)
    stats = {"action": {
        "mean": training_actions.mean(axis=0).tolist(),
        "std": training_actions.std(axis=0).tolist(),
        "q01": np.quantile(training_actions, .01, axis=0).tolist(),
        "q99": np.quantile(training_actions, .99, axis=0).tolist(),
        "min": training_actions.min(axis=0).tolist(),
        "max": training_actions.max(axis=0).tolist()
    }}
    result = {
        "format": "stackcube_openvlaoft_sft_intermediate_v1",
        "source_dir": str(source.resolve()), "control_mode": "pd_ee_delta_pose",
        "cameras": ["base_rgb", "wrist_rgb"],
        "num_action_chunks": chunk, "action_dim": 7,
        "splits": splits,
        "episodes": {"train": [e["episode"] for i,e in enumerate(entries) if i not in val_ids],
                     "val": [e["episode"] for i,e in enumerate(entries) if i in val_ids]}
    }
    (output / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "train_action_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return result


def read_example(index_path, split, sample_id):
    index = json.loads(Path(index_path).read_text())
    sample = index["splits"][split][sample_id]
    t, chunk = sample["step"], index["num_action_chunks"]
    with np.load(Path(index["source_dir"]) / sample["file"], allow_pickle=False) as data:
        return {"base_rgb": data["base_rgb"][t].copy(),
                "wrist_rgb": data["wrist_rgb"][t].copy(),
                "actions": data["actions"][t:t+chunk].astype(np.float32).copy(),
                "instruction": sample["instruction"], "episode": sample["episode"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.input, args.output, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Indexed {len(result['splits']['train'])} training and "
          f"{len(result['splits']['val'])} validation 8-action examples")
    print("No OpenVLA-OFT weights have been trained.")


if __name__ == "__main__":
    main()
