# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read-only SFT/FR3 request-deployment audit before another physical rollout.

Checks dataset lineage, train-only 32D OpenPI quantiles, exact 7D request
channel contract and existing rollout motion evidence. Does NOT approve the
running controller binary, frame convention, timestamp synchronization or
robot safety; those require independent physical validation.

No ROS imports, inference, robot commands, or checkpoint modifications.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from toolkits.realworld_check.fr3_pi05_request_adapter import ARM_AXES
from toolkits.realworld_check.fr3_request_openpi_preflight import validate_metadata
from toolkits.realworld_check.fr3_rollout_motion_diagnostic import diagnose


def audit(train: Path, validation: Path, norm_dir: Path,
          *, episode: Path | None = None) -> dict:
    # Confirm that normalization matches the exact training provenance.
    meta = validate_metadata(train, validation, norm_dir)
    payload = json.loads((norm_dir / "norm_stats.json").read_text())
    stats = payload.get("norm_stats", payload)
    actions = stats["actions"]
    mean = actions["mean"][:7]
    q01, q99 = actions["q01"][:7], actions["q99"][:7]
    for j, (lo, hi, mu) in enumerate(zip(q01, q99, mean)):
        if not all(math.isfinite(float(x)) for x in (lo, hi, mu)) or hi <= lo:
            raise ValueError(f"Bad action quantile at channel {j}")
    if any(float(lo) < -1.001 or float(hi) > 1.001
           for lo, hi in zip(q01[:6], q99[:6])):
        raise ValueError("Training action statistics not normalized SpaceMouse requests")
    report = {
        "mode": "offline_read_only",
        "train_source_episodes": meta["train"]["source_episodes"],
        "validation_source_episodes": meta["validation"]["source_episodes"],
        "train_only_norm_passed": meta["train_only_norm_passed"],
        "source_episode_disjoint": meta["source_episode_disjoint"],
        "decoded_action_7d_contract": [*ARM_AXES, "latched_gripper_open_request"],
        "model_action_dimension": 32,
        "robot_request_dimension": 7,
        "raw_request_train_mean": [round(float(x), 6) for x in mean],
        "raw_request_train_q01": [round(float(x), 6) for x in q01],
        "raw_request_train_q99": [round(float(x), 6) for x in q99],
        "inference_inverse": "OpenPI Unnormalize on train-only actions, then [:, :7]",
        "existing_live_extra_arm_gain": 0.15,
        "proposed_request_mapping_gain": 1.0,
        "trained_action_representation": "time-weighted average normalized SpaceMouse input",
        "controller_consumes": "instantaneous normalized Twist request; NUC C++ mapping",
        "controller_binary_version_verified": False,
        "mounting_frame_and_axis_signs_verified": False,
        "action_hold_time_and_latency_verified": False,
        "gripper_bridge_goal_acceptance_verified": False,
        "physical_identity_execution_approved": False,
        "decision": "software_contract_inspected_physical_controller_verification_pending",
    }
    if episode is not None:
        report["existing_episode"] = diagnose(episode / "policy_command_log.jsonl")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path,
                   default=Path("/data1/Muttaqien/fr3_peg_request_lerobot_train_v1"))
    p.add_argument("--validation", type=Path,
                   default=Path("/data1/Muttaqien/fr3_peg_request_lerobot_validation_v1"))
    p.add_argument("--norm-dir", type=Path,
                   default=Path("/data1/Muttaqien/fr3_peg_request_norm_v1"))
    p.add_argument("--episode", type=Path, help="Optional existing rollout directory")
    a = p.parse_args()
    print(json.dumps(audit(a.train, a.validation, a.norm_dir,
                           episode=a.episode), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
