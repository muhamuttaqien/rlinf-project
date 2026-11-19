# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline observed-motion proxy from an existing FR3 raw episode.

READ ONLY with respect to the source episode. This is a DIAGNOSTIC, not an
accepted-controller action label or a LeRobot/PI0.5/RECAP training dataset.
Its 7D proxies use measured NEXT poses and gripper readings (future leakage);
they must not be passed to fr3_labeled_episode_export or executed on a robot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from toolkits.realworld_check.fr3_raw_capture import RAW_SCHEMA

SCHEMA = "fr3_observed_transition_proxy_v1"
SOURCE = "consecutive_measured_tcp_poses_and_next_measured_knuckle"
FIELDS = (
    "observed_delta_tcp_x_m", "observed_delta_tcp_y_m",
    "observed_delta_tcp_z_m", "observed_delta_tcp_rotvec_x_rad",
    "observed_delta_tcp_rotvec_y_rad", "observed_delta_tcp_rotvec_z_rad",
    "next_measured_gripper_knuckle_rad",
)


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def _quaternion(value: Any) -> np.ndarray:
    quat = _vector(value, 4, "TCP quaternion xyzw")
    norm = float(np.linalg.norm(quat))
    if not 0.95 <= norm <= 1.05:
        raise ValueError(f"TCP quaternion norm {norm:.5f} outside [0.95, 1.05]")
    return quat / norm  # Normalization used only for diagnostic rotation math.


def _multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x, y, z, w = a
    X, Y, Z, W = b
    return np.array([
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
        w * W - x * X - y * Y - z * Z,
    ])


def base_frame_rotvec(q0: Any, q1: Any) -> np.ndarray:
    """Shortest axis-angle of q1 * inverse(q0); assumes same base frame."""
    q0 = _quaternion(q0)
    q1 = _quaternion(q1)
    dq = _multiply_xyzw(q1, np.array([-q0[0], -q0[1], -q0[2], q0[3]]))
    if dq[3] < 0:
        dq = -dq  # Remove q / -q representation discontinuity.
    vnorm = float(np.linalg.norm(dq[:3]))
    if vnorm < 1e-12:
        return 2.0 * dq[:3]
    angle = 2.0 * math.atan2(vnorm, max(0.0, float(dq[3])))
    return dq[:3] * (angle / vnorm)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    low = math.floor(rank)
    high = math.ceil(rank)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def _timestamp(record: dict[str, Any]) -> int:
    value = record.get("recorded_monotonic_ns")
    if type(value) is not int or value <= 0:
        raise ValueError("Missing positive workstation monotonic timestamp")
    return value


def _source_stamp(record: dict[str, Any], name: str) -> int | None:
    data = record.get("source_and_camera_metadata", {}).get("source", {})
    entry = data.get(name, {})
    stamp = entry.get("received_monotonic_ns")
    return stamp if type(stamp) is int and stamp > 0 else None


def analyze_episode(raw: Path, *, min_dt_s: float = 0.01,
                    max_dt_s: float = 0.5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Analyze observed frame-to-frame transitions; no inferred controller targets."""
    raw = Path(raw)
    if not 0 < min_dt_s < max_dt_s or not all(
        math.isfinite(v) for v in (min_dt_s, max_dt_s)
    ):
        raise ValueError("Invalid minimum/maximum interval")
    manifest_path = raw / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != RAW_SCHEMA or manifest.get("status") != "raw_recorded":
        raise ValueError("Expected one completed FR3 raw capture")
    with (raw / "states.jsonl").open(encoding="utf-8") as stream:
        frames = [json.loads(line) for line in stream if line.strip()]
    if len(frames) < 2 or len(frames) != manifest.get("frames_saved"):
        raise ValueError("Raw state count must match manifest and exceed 1")
    if any(record.get("training_action") is not None for record in frames):
        raise ValueError("Raw episode unexpectedly contains training action labels")

    positions = []
    quaternions = []
    grippers = []
    stamps = []
    source_stamps = []
    for index, record in enumerate(frames):
        if record.get("frame_index") != index:
            raise ValueError(f"Unexpected frame index {index}")
        state = record["measured_state"]
        pose = _vector(state["tcp_pose"], 7, "TCP pose xyz + xyzw")
        positions.append(pose[:3])
        quaternions.append(_quaternion(pose[3:]))
        grippers.append(float(_vector(state["robotiq_knuckle_position"], 1,
                                      "Robotiq knuckle angle")[0]))
        stamps.append(_timestamp(record))
        source_stamps.append(_source_stamp(record, "tcp_pose"))

    records = []
    good_dts: list[float] = []
    all_dts: list[float] = []
    rejected_intervals = 0
    moving_intervals = 0
    for i in range(len(frames) - 1):
        dt_s = (stamps[i + 1] - stamps[i]) / 1e9
        if math.isfinite(dt_s):
            all_dts.append(dt_s)
        if not min_dt_s <= dt_s <= max_dt_s:
            rejected_intervals += 1
            continue
        delta_xyz = positions[i + 1] - positions[i]
        delta_rot = base_frame_rotvec(quaternions[i], quaternions[i + 1])
        proxy = np.concatenate((delta_xyz, delta_rot, [grippers[i + 1]]))
        if not np.isfinite(proxy).all():
            raise ValueError(f"Nonfinite proxy at transition {i}")
        if np.linalg.norm(delta_xyz) >= 0.0002 or np.linalg.norm(delta_rot) >= 0.002:
            moving_intervals += 1
        good_dts.append(dt_s)
        records.append({
            "frame_index": i,
            "next_frame_index": i + 1,
            "observation_monotonic_ns": stamps[i],
            "next_observation_monotonic_ns": stamps[i + 1],
            "dt_s": dt_s,
            "observed_transition_proxy_7d": proxy.tolist(),
        })

    if not records:
        raise ValueError("No valid adjacent timestamp intervals; no proxies generated")
    observed_sources = [s for s in source_stamps if s is not None]
    repeated_source_stamps = sum(
        a == b for a, b in zip(source_stamps, source_stamps[1:])
        if a is not None and b is not None
    )
    gripper_changes = sum(
        abs(a - b) > 1e-4 for a, b in zip(grippers, grippers[1:])
    )
    summary = {
        "schema": SCHEMA,
        "raw_episode": raw.name,
        "raw_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "frames": len(frames),
        "observed_transition_proxies": len(records),
        "rejected_time_intervals": rejected_intervals,
        "median_dt_s": median(good_dts),
        "p95_dt_s": _percentile(good_dts, 0.95),
        "min_observed_dt_s": min(all_dts),
        "max_observed_dt_s": max(all_dts),
        "moving_intervals": moving_intervals,
        "gripper_change_intervals": gripper_changes,
        "tcp_source_stamps_present": len(observed_sources),
        "repeated_tcp_source_stamps": repeated_source_stamps,
        "action_source": SOURCE,
        "action_fields": list(FIELDS),
        "orientation_delta": "base-frame shortest rotvec q_next * inverse(q_current)",
        "gripper_channel": "next MEASURED knuckle angle in radians (NOT controller target)",
        "last_frame": "no forward transition; intentionally unlabeled",
        "is_controller_accepted_action": False,
        "is_lerobot_dataset": False,
        "sft_ready": False,
        "recap_ready": False,
    }
    return summary, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="Optional NEW analysis JSON outside the original session")
    parser.add_argument("--min-dt-s", type=float, default=0.01)
    parser.add_argument("--max-dt-s", type=float, default=0.5)
    args = parser.parse_args()

    summary, records = analyze_episode(
        args.raw, min_dt_s=args.min_dt_s, max_dt_s=args.max_dt_s)
    if args.output is not None:
        raw_session = args.raw.expanduser().resolve().parent
        output = args.output.expanduser().resolve()
        if output.is_relative_to(raw_session):
            parser.error("--output must be OUTSIDE the raw capture session directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump({"schema": SCHEMA, "summary": summary, "records": records},
                      stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        f"{summary['raw_episode']}: {summary['frames']} frames, "
        f"{summary['observed_transition_proxies']} proxy intervals, "
        f"{summary['rejected_time_intervals']} rejected time intervals"
    )
    print(
        f"Timing: median {summary['median_dt_s']:.3f}s, "
        f"p95 {summary['p95_dt_s']:.3f}s; "
        f"repeated TCP source timestamps {summary['repeated_tcp_source_stamps']}"
    )
    print(
        f"Motion intervals: {summary['moving_intervals']}; "
        f"gripper-change intervals: {summary['gripper_change_intervals']}"
    )
    if args.output is not None:
        print(f"Diagnostic proxies saved: {args.output}")
    print("NOT SFT-READY: observed-motion proxies are not controller commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
