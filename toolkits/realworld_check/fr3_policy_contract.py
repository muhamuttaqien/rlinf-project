# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Milestone 4C.2: explicit PROPOSED 19D FR3 state, unapproved 7D action contract.

Offline only. This module cannot produce 7D action labels, LeRobot datasets,
policy-ready observations, controller messages or robot commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from toolkits.realworld_check.fr3_lerobot_preflight import audit_raw_episode

CONTRACT_VERSION = "fr3_robotiq_policy_contract_candidate_v2"
FEATURE_NAMES = (
    "tcp_x_m", "tcp_y_m", "tcp_z_m",
    "tcp_quat_x", "tcp_quat_y", "tcp_quat_z", "tcp_quat_w",
    "fr3_joint1_rad", "fr3_joint2_rad", "fr3_joint3_rad",
    "fr3_joint4_rad", "fr3_joint5_rad", "fr3_joint6_rad", "fr3_joint7_rad",
    "wrench_force_x_N", "wrench_force_y_N", "wrench_force_z_N",
    "robotiq_left_knuckle_rad",
    "wrench_torque_magnitude_Nm",
)
ACTION_FIELDS = (
    "delta_tcp_x_m", "delta_tcp_y_m", "delta_tcp_z_m",
    "delta_tcp_rotvec_x_rad", "delta_tcp_rotvec_y_rad",
    "delta_tcp_rotvec_z_rad", "gripper_target_knuckle_rad",
)


def _vector(state: Mapping[str, Any], field: str, dim: int) -> np.ndarray:
    if field not in state:
        raise ValueError(f"Missing measured state: {field}")
    try:
        array = np.asarray(state[field], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if array.shape != (dim,) or not np.isfinite(array).all():
        raise ValueError(f"{field} must be {dim} finite scalars")
    return array


def candidate_19d_state(measured_state: Mapping[str, Any]) -> np.ndarray:
    """Construct a review-only 19D observation from recorded physical readings.

    Explicit dimensional choice: TCP pose (7), measured joint position (7),
    measured force (3), Robotiq knuckle angle (1), wrench torque norm (1).
    This is *not* the existing nested Franka 20D vector with a component
    silently removed, nor a calibrated policy state.
    """
    if not isinstance(measured_state, Mapping):
        raise ValueError("Measured state must be a mapping")
    pose = _vector(measured_state, "tcp_pose", 7)
    joints = _vector(measured_state, "arm_joint_position", 7)
    force = _vector(measured_state, "tcp_force", 3)
    knuckle = _vector(measured_state, "robotiq_knuckle_position", 1)
    torque = _vector(measured_state, "tcp_torque", 3)
    norm = float(np.linalg.norm(pose[3:]))
    if not 0.95 <= norm <= 1.05:
        raise ValueError("TCP quaternion is not approximately unit length")
    # Do not normalize silently: preserving the measured source is essential.
    result = np.concatenate((
        pose, joints, force, knuckle, [float(np.linalg.norm(torque))]
    )).astype(np.float32)
    if result.shape != (19,) or not np.isfinite(result).all():
        raise ValueError("Candidate state is not 19 finite scalars")
    return result


def candidate_contract_report(raw: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Audit the full raw episode, then preview a single explicitly named state."""
    prior, _ = audit_raw_episode(raw)
    with (Path(raw) / "states.jsonl").open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
    state = candidate_19d_state(first["measured_state"])
    sources = first["source_and_camera_metadata"].get("source", {})

    def frame_id(topic: str) -> str | None:
        header = sources.get(topic, {}).get("header") or {}
        return header.get("frame_id")

    return {
        "contract_version": CONTRACT_VERSION,
        "status": "candidate_requires_review",
        "raw_capture_valid": prior["raw_capture_valid"],
        "raw_sample_count": prior["sample_count"],
        "state_dim": 19,
        "state_feature_names": list(FEATURE_NAMES),
        "state_preview_dtype": "float32",
        "state_mapping": "tcp_pose7 + joints_position7 + force_xyz3 + knuckle_angle1 + torque_l2_norm1",
        "state_source_frames": {
            "tcp_pose": frame_id("tcp_pose"),
            "external_wrench": frame_id("external_wrench"),
        },
        "state_tradeoffs": [
            "Torque vector direction is replaced by its magnitude (Nm).",
            "Joint velocities and efforts are omitted deliberately, not truncated.",
            "TCP frame/tool offset, gripper width calibration and cross-domain timing require verification.",
        ],
        "action_dim_proposed": 7,
        "action_fields_proposed": list(ACTION_FIELDS),
        "gripper_bridge_reference": {
            "source": "NUC /tmp/robotiq_cross_domain_bridge.py (operator inspected 2026-11-07)",
            "action": "/robotiq_gripper_controller/gripper_cmd",
            "action_type": "control_msgs/action/GripperCommand",
            "open_request": 1.0,
            "open_position_target": 0.0,
            "close_request": 0.0,
            "close_position_target": 0.7929,
            "max_effort_field": 50.0,
            "goal_queue_maxsize": 1,
            "is_continuous_width_controller": False,
            "goal_unit_and_joint_mapping_verified": False,
        },
        "action_translation_frame": "robot_base_FRAME_UNVERIFIED",
        "action_rotation": "rotation_vector_axis_angle_FRAME_UNVERIFIED",
        "action_interval_s": None,
        "executed_action_source": None,
        "gripper_width_calibration_verified": False,
        "action_labels_available": False,
        "le_robot_export_ready": False,
        "sft_ready": False,
        "recap_ready": False,
        "required_approvals": [
            "Approve or revise this exact 19D state feature selection.",
            "Verify frame IDs, TCP/tool offset and the quaternion xyzw convention.",
            "Verify Robotiq controller target position units/joint and safe limits; do not infer metric aperture width.",
            "Verify provenance and timing of accepted EE and Robotiq 7D controller goals; normalized operator input is not a target.",
            "Collect human-labeled task demonstration episodes after physical safety clearance.",
            "Verify OpenPI SFT and RECAP two-camera loader compatibility.",
        ],
    }, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path,
        help="Optional NEW directory for proposal report and first-state preview",
    )
    args = parser.parse_args()
    report, first_state = candidate_contract_report(args.raw)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "contract_candidate.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        np.save(args.output / "state_candidate_first.npy", first_state, allow_pickle=False)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
