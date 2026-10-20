# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Audit StackCube OpenVLA-OFT supervised-training readiness.

The generic RLinf VLA SFT worker does not currently support openvla_oft.
The upstream OpenVLA-OFT trainer requires an RLDS dataset and a matching
action normalization key. This check never downloads weights or starts training.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(index_path, stats_path):
    index = json.loads(Path(index_path).read_text())
    stats = json.loads(Path(stats_path).read_text())["action"]
    if index.get("format") != "stackcube_openvlaoft_sft_intermediate_v1":
        raise ValueError("Unexpected intermediate dataset format")
    if index["num_action_chunks"] != 8 or index["action_dim"] != 7:
        raise ValueError("Expected eight 7D actions per example")
    train = set(index["episodes"]["train"])
    val = set(index["episodes"]["val"])
    if not train or not val or train & val:
        raise ValueError("Training and validation episodes must not overlap")
    for name in ("mean", "std", "q01", "q99", "min", "max"):
        array = np.asarray(stats[name], dtype=float)
        if array.shape != (7,) or not np.all(np.isfinite(array)):
            raise ValueError(f"Invalid action statistic: {name}")
    result = {
        "dataset": str(index_path), "train_episodes": len(train),
        "validation_episodes": len(val),
        "train_examples": len(index["splits"]["train"]),
        "validation_examples": len(index["splits"]["val"]),
        "action_mode": index.get("control_mode"),
        "action_dim": 7, "chunk": 8,
        "rlinf_generic_sft_worker_supports_openvla_oft": False,
        "upstream_openvla_oft_requires_rlds": True,
        "rlds_dataset_created": False,
        "checkpoint_action_stats_validated": False,
        "policy_training_started": False,
        "next_steps": [
            "Convert episode-separated StackCube recordings to TensorFlow Datasets RLDS",
            "Register StackCube in the OpenVLA-OFT RLDS OXE dataset configuration",
            "Verify pd_ee_delta_pose rotation, gripper semantics and checkpoint normalization",
            "Install and run upstream OpenVLA-OFT LoRA trainer with a validated checkpoint"
        ]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path,
                        default=Path("data/stackcube_openvlaoft_sft/index.json"))
    parser.add_argument("--stats", type=Path,
                        default=Path("data/stackcube_openvlaoft_sft/train_action_stats.json"))
    args = parser.parse_args()
    print(json.dumps(audit(args.index, args.stats), indent=2))


if __name__ == "__main__":
    main()
