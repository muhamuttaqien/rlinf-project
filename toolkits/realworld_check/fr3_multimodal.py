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

"""Validate and assemble offline FR3, Robotiq and two-camera observations."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
SCHEMA = "fr3_robotiq_multimodal_raw_v1"


def _source(snapshot: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Reject missing or unavailable source measurements."""
    state = snapshot.get("states", {}).get(name, {})
    if state.get("status") != "available" or not isinstance(
        state.get("source"), Mapping
    ):
        raise ValueError(f"{name} unavailable: {state.get('reason', 'missing')}")
    source = state["source"]
    if not isinstance(source.get("message"), Mapping):
        raise ValueError(f"{name} has no ROS message")
    return source


def _numbers(values: Any, length: int, name: str) -> np.ndarray:
    """Validate one numeric measurement without padding or truncation."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have {length} finite values; got {array.shape}")
    return array.astype(np.float32)


def _xyz(record: Mapping[str, Any], key: str, name: str) -> np.ndarray:
    try:
        values = [record[key][axis] for axis in ("x", "y", "z")]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{name} must contain x, y and z") from exc
    return _numbers(values, 3, name)


def _provenance(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "topic": source.get("topic"),
        "message_type": source.get("message_type"),
        "header": copy.deepcopy(source["message"].get("header")),
        "received_monotonic_ns": source.get("received_monotonic_ns"),
        "received_unix_ns": source.get("received_unix_ns"),
    }


def assemble_fr3_multimodal(
    arm_snapshot: Mapping[str, Any],
    gripper_snapshot: Mapping[str, Any],
    cameras: Mapping[str, Mapping[str, Any]],
    *,
    max_camera_receipt_skew_s: float = 0.1,
) -> dict[str, Any]:
    """Produce an RLinf raw Observation without controlling any hardware.

    Snapshots are the output of RosStateSubscriber.snapshot() from each ROS
    domain. Camera records match FrankaObservation.get_observation()["cameras"].
    This is not a 19D OpenPI state projection or a timing calibration.
    """
    if not np.isfinite(max_camera_receipt_skew_s) or max_camera_receipt_skew_s <= 0:
        raise ValueError("max_camera_receipt_skew_s must be finite and positive")

    joints = _source(arm_snapshot, "joint_state")
    tcp = _source(arm_snapshot, "tcp_pose")
    wrench = _source(arm_snapshot, "external_wrench")
    gripper = _source(gripper_snapshot, "gripper_state")

    joint_msg = joints["message"]
    names = joint_msg.get("name")
    if not isinstance(names, (list, tuple)) or len(names) != len(set(names)):
        raise ValueError("FR3 joint names must form a unique sequence")
    missing = [name for name in ARM_JOINTS if name not in names]
    if missing:
        raise ValueError(f"Missing FR3 joints: {missing}")
    order = [names.index(name) for name in ARM_JOINTS]

    def joint_values(field: str) -> np.ndarray:
        return _numbers(joint_msg.get(field), len(names), f"joint_state.{field}")[order].copy()

    pose = tcp["message"].get("pose")
    if not isinstance(pose, Mapping) or not isinstance(pose.get("orientation"), Mapping):
        raise ValueError("tcp_pose must contain a pose and quaternion")
    position = _xyz(pose, "position", "tcp_pose.position")
    try:
        quaternion = _numbers(
            [pose["orientation"][axis] for axis in ("x", "y", "z", "w")],
            4, "tcp_pose.quaternion"
        )
    except KeyError as exc:
        raise ValueError("tcp_pose.quaternion must contain xyzw") from exc
    if np.linalg.norm(quaternion) < 1e-6:
        raise ValueError("tcp_pose.quaternion must be nonzero")

    wrench_msg = wrench["message"].get("wrench")
    if not isinstance(wrench_msg, Mapping):
        raise ValueError("external_wrench.wrench missing")
    force = _xyz(wrench_msg, "force", "external_wrench.force")
    torque = _xyz(wrench_msg, "torque", "external_wrench.torque")

    grip_msg = gripper["message"]
    grip_names = grip_msg.get("name")
    if not isinstance(grip_names, (list, tuple)) or len(grip_names) != len(set(grip_names)):
        raise ValueError("Robotiq joint names must form a unique sequence")
    if GRIPPER_JOINT not in grip_names:
        raise ValueError(f"Missing Robotiq joint: {GRIPPER_JOINT}")
    grip_positions = _numbers(
        grip_msg.get("position"), len(grip_names), "gripper_state.position"
    )
    gripper_angle = grip_positions[grip_names.index(GRIPPER_JOINT)].reshape(1).copy()

    if set(cameras) != {"base", "wrist"}:
        raise ValueError("Exactly base and wrist camera roles are required")
    frames = {}
    rgb_frames = {}
    camera_info = {}
    serials = set()
    timestamps = []
    for role in ("base", "wrist"):
        camera = cameras[role]
        frame = np.asarray(camera.get("frame"))
        if (
            frame.ndim != 3
            or frame.shape[-1] != 3
            or min(frame.shape[:2]) < 1
            or frame.dtype != np.uint8
        ):
            raise ValueError(f"{role} must contain HWC uint8 3-channel image")
        if camera.get("encoding") != "bgr8":
            raise ValueError(f"{role} encoding must explicitly be bgr8")
        serial = camera.get("serial_number")
        if not isinstance(serial, str) or not serial or serial in serials:
            raise ValueError("Camera serials must be distinct and nonempty")
        serials.add(serial)
        stamp = camera.get("received_monotonic_ns")
        if not isinstance(stamp, int) or stamp <= 0:
            raise ValueError(f"{role} must have positive host receipt time")
        timestamps.append(stamp)
        frames[role] = frame.copy()
        rgb_frames[role] = np.ascontiguousarray(frame[..., ::-1])
        camera_info[role] = {
            "serial_number": serial,
            "raw_encoding": "bgr8",
            "rgb_encoding": "rgb8",
            "received_monotonic_ns": stamp,
            "timestamp_kind": camera.get("timestamp_kind"),
        }
    skew = abs(timestamps[0] - timestamps[1]) / 1e9
    if skew > max_camera_receipt_skew_s:
        raise ValueError(f"Camera receipt skew {skew:.4f}s exceeds limit")

    return {
        "state": {
            "arm_joint_position": joint_values("position"),
            "arm_joint_velocity": joint_values("velocity"),
            "arm_joint_effort": joint_values("effort"),
            "tcp_pose": np.concatenate((position, quaternion)),
            "tcp_force": force,
            "tcp_torque": torque,
            "robotiq_knuckle_position": gripper_angle,
        },
        "frames": frames,
        "policy_images_rgb": rgb_frames,
        "metadata": {
            "schema": SCHEMA,
            "policy_ready": False,
            "arm_joint_names": list(ARM_JOINTS),
            "gripper_joint_name": GRIPPER_JOINT,
            "gripper_units": "knuckle_radians_not_metric_width",
            "source": {
                "joint_state": _provenance(joints),
                "tcp_pose": _provenance(tcp),
                "external_wrench": _provenance(wrench),
                "gripper_state": _provenance(gripper),
            },
            "cameras": camera_info,
            "camera_receipt_skew_s": skew,
            "clock_alignment": "unverified_across_hosts",
        },
    }
