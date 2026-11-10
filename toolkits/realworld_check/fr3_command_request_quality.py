# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline quality audit and episode-level split of FR3 command-request CANDIDATES.

Consumes existing command_request_candidate_v1 JSONL. It does NOT fill missing
gripper requests, copy images, create LeRobot data, train models, launch ROS,
or send robot commands. A split manifest is NOT an approved SFT dataset:
the 7D labels contain time-averaged normalized SpaceMouse REQUESTS.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

SCHEMA = "fr3_spacemouse_request_quality_v1"
INPUT_SCHEMA = "fr3_spacemouse_request_alignment_candidate_v1"


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{number}: expected JSON object")
                yield item


def _vector(values: object, count: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == count
        and all(
            type(x) in (int, float) and math.isfinite(x)
            for x in values
        )
    )


def _episode_quality(root: Path, source: dict, min_coverage: float,
                     horizon: int) -> dict:
    name = source["episode"]
    if not isinstance(name, str) or not name.startswith("episode_") or "/" in name:
        raise ValueError("Invalid episode identifier")
    file = root / "episodes" / (name + ".jsonl")
    counters = Counter()
    spans = []
    span_start = None
    first_gripper_frame = None
    seen_gripper = False
    prior_frame = None
    n = 0

    for row in _read_rows(file):
        frame = row.get("frame_index")
        if type(frame) is not int or frame != n or row.get("next_frame_index") != n + 1:
            raise ValueError(f"{name}: noncontiguous or reordered frame indices")
        n += 1
        dt = row.get("observation_dt_s")
        coverage = row.get("twist_request_time_coverage")
        if type(dt) not in (int, float) or not math.isfinite(dt):
            raise ValueError(f"{name}: invalid observation interval")
        if type(coverage) not in (int, float) or not 0 <= coverage <= 1:
            raise ValueError(f"{name}: invalid coverage")
        dt_ok = 0.02 <= dt <= 0.25
        arm = row.get("spacemouse_mean_request_6d")
        arm_valid = dt_ok and coverage >= min_coverage and _vector(arm, 6)
        grip = row.get("gripper_request_open_fraction")
        if grip is not None and (type(grip) not in (int, float) or grip not in (0, 1)):
            raise ValueError(f"{name}: invalid normalized gripper request")
        grip_known = grip is not None
        if not grip_known:
            counters["unknown_gripper_request"] += 1
            if seen_gripper:
                counters["unknown_gripper_after_first_request"] += 1
            else:
                counters["initial_unknown_gripper_prefix"] += 1
        else:
            if not seen_gripper:
                first_gripper_frame = frame
            seen_gripper = True

        if not dt_ok:
            counters["invalid_dt"] += 1
        elif coverage < min_coverage:
            counters["insufficient_twist_coverage"] += 1
        if not arm_valid:
            counters["incomplete_6d_twist"] += 1
            if not grip_known:
                counters["twist_problem_overlaps_unknown_gripper"] += 1
        else:
            counters["complete_6d_twist"] += 1

        complete = arm_valid and grip_known
        action = row.get("candidate_request_action_7d")
        if row.get("candidate_complete") is not complete:
            raise ValueError(f"{name} frame {frame}: inconsistent completion flag")
        if complete:
            if not _vector(action, 7) or any(abs(a-b) > 1e-5 for a,b in
                                                    zip(action, [*arm, grip])):
                raise ValueError(f"{name} frame {frame}: mismatched candidate action")
            counters["complete_7d_request"] += 1
            if span_start is None:
                span_start = frame
        else:
            if action is not None:
                raise ValueError(f"{name} frame {frame}: incomplete row has action")
            if span_start is not None:
                spans.append([span_start, frame])
                span_start = None
    if span_start is not None:
        spans.append([span_start, n])

    if n != source.get("transitions"):
        raise ValueError(f"{name}: transition count differs from source report")
    if (counters["complete_7d_request"]
            != source.get("counts", {}).get("candidate_7d_available", 0)):
        raise ValueError(f"{name}: complete-action count differs from source report")
    if counters["unknown_gripper_after_first_request"]:
        raise ValueError(f"{name}: gripper request became unknown after first request")
    windows = sum(max(0, end-start-horizon+1) for start, end in spans)
    return {
        "episode": name,
        "source_jsonl": str(file),
        "transitions": n,
        "first_known_gripper_frame": first_gripper_frame,
        "valid_spans_end_exclusive": spans,
        "complete_7d_horizon_windows": windows,
        "counts": dict(counters),
    }


def audit(root: Path, *, horizon: int = 10, validation_fraction: float = 0.2,
          seed: int = 42) -> dict:
    root = Path(root).expanduser().resolve()
    if horizon < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Invalid split or horizon settings")
    source = _read_json(root / "report.json")
    if source.get("schema") != INPUT_SCHEMA:
        raise ValueError("Not an FR3 command-request candidate dataset")
    reports = source.get("episode_reports")
    if not isinstance(reports, list) or len(reports) < 2:
        raise ValueError("At least two source episodes required for split")
    seen = set()
    episode_results = []
    for ep in reports:
        name = ep.get("episode")
        if name in seen:
            raise ValueError(f"Duplicate episode: {name}")
        seen.add(name)
        episode_results.append(_episode_quality(
            root, ep, source["min_twist_time_coverage"], horizon,
        ))

    counts = Counter()
    for entry in episode_results:
        counts.update(entry["counts"])
    if sum(x["transitions"] for x in episode_results) != source.get("transitions"):
        raise ValueError("Global transition count differs from source report")
    if counts["complete_7d_request"] != source.get("candidate_7d_available"):
        raise ValueError("Global complete-action count differs from source report")

    names = sorted(seen)
    shuffled = names[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, min(len(names)-1, round(len(names) * validation_fraction)))
    validation = set(shuffled[:n_val])
    for entry in episode_results:
        entry["split"] = "validation" if entry["episode"] in validation else "train"
    train = [x for x in episode_results if x["split"] == "train"]
    val = [x for x in episode_results if x["split"] == "validation"]

    def split_summary(items):
        return {
            "episodes": len(items),
            "complete_7d_transitions": sum(
                x["counts"].get("complete_7d_request", 0) for x in items
            ),
            "contiguous_horizon_windows": sum(
                x["complete_7d_horizon_windows"] for x in items
            ),
        }

    return {
        "schema": SCHEMA,
        "source_candidate_root": str(root),
        "source_report": str(root / "report.json"),
        "horizon": horizon,
        "validation_fraction": validation_fraction,
        "split_seed": seed,
        "split_unit": "whole_episode_no_cross_episode_leakage",
        "counts": dict(counts),
        "train": split_summary(train),
        "validation": split_summary(val),
        "episode_manifests": episode_results,
        "gripper_unknown_imputed": False,
        "arm_values_are_time_averaged_normalized_requests": True,
        "valid_spans_keep_original_frame_indices": True,
        "source_images_and_dataset_unchanged": True,
        "lerobot_exported": False,
        "sft_ready": False,
        "recap_ready": False,
        "robot_deployment_approved": False,
        "next_gate": (
            "Verify controller command timing, frames and gripper mapping. "
            "Then define a new OpenPI request-action transform and normalizer. "
            "Never reuse proxy-action normalization/checkpoint for this dataset."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.candidate_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == root or output.is_relative_to(root):
        parser.error("--output must be OUTSIDE original candidate dataset")
    if output.exists():
        parser.error("Refusing to overwrite existing output")
    result = audit(root, horizon=args.horizon,
                   validation_fraction=args.validation_fraction, seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    c = result["counts"]
    print(f"Complete 7D requests: {c.get('complete_7d_request', 0)}")
    print(f"Unknown gripper prefix: {c.get('initial_unknown_gripper_prefix', 0)}")
    print(f"Twist gaps overlapping unknown gripper: "
          f"{c.get('twist_problem_overlaps_unknown_gripper', 0)}")
    print(f"Train: {result['train']}")
    print(f"Validation: {result['validation']}")
    print(f"Audit manifest: {output}")
    print("NOT a LeRobot / SFT / RECAP or robot-ready dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
