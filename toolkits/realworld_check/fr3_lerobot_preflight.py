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

"""Read-only 4C preflight for raw FR3 episodes; NEVER export false training labels.

This validates persisted observations and writes a two-view RGB preview.
It is not a LeRobot exporter: no 19D state mapping, 7D training actions,
demonstration-success labels, normalization, or episode windows are invented.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from toolkits.realworld_check.fr3_raw_capture import RAW_SCHEMA

STATE_DIMENSIONS = {
    "arm_joint_position": 7,
    "arm_joint_velocity": 7,
    "arm_joint_effort": 7,
    "tcp_pose": 7,
    "tcp_force": 3,
    "tcp_torque": 3,
    "robotiq_knuckle_position": 1,
}
ROLES = ("base", "wrist")
BLOCKERS = [
    "No approved 19D OpenPI policy state projection (raw fields are not a policy state).",
    "No verified metric 7D action labels or executed-controller action alignment.",
    "No human-reviewed moving-task episodes with terminal/success labels.",
    "Cross-host clock/exposure synchronization and action-time alignment unverified.",
    "RECAP FrankaEEInputs does not match the required two-camera RealworldInputs path.",
]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing {path.name}")
    result = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path.name} contains blank line {index}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} line {index} is not an object")
        result.append(value)
    return result


def _finite_vector(value: object, count: int, label: str) -> None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain {count} finite values")


def _positive_stamp(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer timestamp")
    return value


def _frame(root: Path, role: str, index: int, record: dict, profile: dict) -> np.ndarray:
    name = f"frames/{role}/{index:06d}.npy"
    if record.get("image_files_bgr8", {}).get(role) != name:
        raise ValueError(f"Frame {index}: {role} path differs from expected {name}")
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or linked {name}")
    frame = np.load(path, allow_pickle=False, mmap_mode="r")
    width, height = profile["resolution"]
    if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
        raise ValueError(f"{name}: expected {(height, width, 3)} uint8 BGR")
    return frame


def audit_raw_episode(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Validate every record and file; return report and first real RGB pair."""
    root = Path(root)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema") != RAW_SCHEMA:
        raise ValueError("Unexpected raw capture schema")
    if manifest.get("status") != "raw_recorded":
        raise ValueError("Only fully recorded raw episodes qualify for 4C preflight")
    for flag in ("is_lerobot_dataset", "policy_ready", "action_targets_7d_available",
                 "executed_arm_commands_available"):
        if manifest.get(flag) is not False:
            raise ValueError(f"Raw capture flag {flag} must be false")
    if manifest.get("operator_events_dropped") != 0 or manifest.get("failures"):
        raise ValueError("Raw capture has dropped events or reported failures")
    if not isinstance(manifest.get("task"), str) or not manifest["task"].strip():
        raise ValueError("Missing task text")

    profiles = manifest.get("camera_profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(ROLES):
        raise ValueError("Expected exactly base and wrist camera profiles")
    serials = []
    for role in ROLES:
        prof = profiles[role]
        if not isinstance(prof, dict):
            raise ValueError(f"{role} profile missing")
        resolution = prof.get("resolution")
        if (not isinstance(resolution, list) or len(resolution) != 2
                or any(type(x) is not int or x <= 0 for x in resolution)):
            raise ValueError(f"{role} resolution invalid")
        serial = prof.get("serial_number")
        if not isinstance(serial, str) or not serial:
            raise ValueError(f"{role} serial missing")
        serials.append(serial)
    if serials[0] == serials[1]:
        raise ValueError("Both cameras have the same serial")

    states = _rows(root / "states.jsonl")
    events = _rows(root / "operator_commands.jsonl")
    if not states or len(states) != manifest.get("frames_saved"):
        raise ValueError("State count does not match manifest")
    if len(events) != manifest.get("operator_command_events_saved"):
        raise ValueError("Event count does not match manifest")

    first_rgb = {}
    previous_stamp = 0
    for i, record in enumerate(states):
        if record.get("frame_index") != i:
            raise ValueError(f"Nonconsecutive frame_index at row {i}")
        stamp = _positive_stamp(record.get("recorded_monotonic_ns"), "recorded time")
        if stamp <= previous_stamp:
            raise ValueError("Nonmonotonic observation receipt times")
        previous_stamp = stamp
        state = record.get("measured_state")
        if not isinstance(state, dict) or set(state) != set(STATE_DIMENSIONS):
            raise ValueError(f"Frame {i}: missing/unexpected raw state fields")
        for name, count in STATE_DIMENSIONS.items():
            _finite_vector(state[name], count, f"Frame {i} {name}")
        if record.get("training_action") is not None:
            raise ValueError("Raw capture unexpectedly contains training actions")
        metadata = record.get("source_and_camera_metadata")
        if not isinstance(metadata, dict) or metadata.get("policy_ready") is not False:
            raise ValueError(f"Frame {i}: invalid raw metadata")
        cameras = metadata.get("cameras")
        if not isinstance(cameras, dict) or set(cameras) != set(ROLES):
            raise ValueError(f"Frame {i}: missing camera provenance")
        for role in ROLES:
            camera = cameras[role]
            if (camera.get("raw_encoding") != "bgr8"
                    or camera.get("serial_number") != profiles[role]["serial_number"]):
                raise ValueError(f"Frame {i}: {role} encoding/serial mismatch")
            _positive_stamp(camera.get("received_monotonic_ns"), f"{role} camera time")
            frame = _frame(root, role, i, record, profiles[role])
            if i == 0:
                first_rgb["image" if role == "base" else "extra_view_image"] = (
                    np.ascontiguousarray(frame[..., ::-1])
                )
    for role in ROLES:
        disk = sorted((root / "frames" / role).glob("*.npy"))
        if len(disk) != len(states):
            raise ValueError(f"Unexpected frame file count for {role}")

    event_counts = {"operator_twist": 0, "operator_gripper_request": 0}
    for i, event in enumerate(events):
        name = event.get("name")
        if name not in event_counts:
            raise ValueError(f"Event {i}: unexpected source name")
        _positive_stamp(event.get("received_monotonic_ns"), f"event {i} receipt")
        event_counts[name] += 1

    report = {
        "audit_schema": "fr3_4c_raw_preflight_v1",
        "source": str(root.resolve()),
        "raw_capture_valid": True,
        "sample_count": len(states),
        "operator_event_count": len(events),
        "operator_event_types": event_counts,
        "camera_serials": dict(zip(ROLES, serials)),
        "camera_resolutions_wh": {r: profiles[r]["resolution"] for r in ROLES},
        "preview_keys": ["image", "extra_view_image"],
        "preview_color": "rgb8 (converted from lossless source bgr8)",
        "task": manifest["task"],
        "is_lerobot_dataset": False,
        "sft_export_ready": False,
        "recap_export_ready": False,
        "training_action_labels_available": False,
        "blocking_requirements": list(BLOCKERS),
    }
    return report, first_rgb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="Existing raw episode directory")
    parser.add_argument(
        "--output", type=Path,
        help="Optional NEW directory for JSON audit and a first-frame RGB preview",
    )
    args = parser.parse_args()
    report, preview = audit_raw_episode(args.raw)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        np.savez_compressed(args.output / "observation_preview_rgb.npz", **preview)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
