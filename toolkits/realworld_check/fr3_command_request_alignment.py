# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Read-only FR3 SpaceMouse-request / 10 Hz observation alignment.

Creates a separate, provenance-rich JSONL candidate dataset referencing the
original RGB .npy files. No ROS, controller commands, LeRobot export, inference,
SFT/RECAP training or physical robot motion. The six arm channels are
NORMALIZED SpaceMouse/Twist REQUESTS, NOT Cartesian position/velocity deltas,
IK commands, accepted torques or executed actions. A seventh channel is the
last recorded normalized gripper REQUEST, or NULL until one is observed.

This is a diagnostic before defining and validating a NEW OpenPI action
contract. The old measured-motion proxy SFT config/norm stats are INCOMPATIBLE.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from toolkits.realworld_check.fr3_policy_contract import (
    CONTRACT_VERSION, candidate_19d_state,
)
from toolkits.realworld_check.fr3_raw_capture import (
    GRIPPER_REQUEST_TOPIC, RAW_SCHEMA, TWIST_TOPIC,
)

SCHEMA = "fr3_spacemouse_request_alignment_candidate_v1"
FIELDS = (
    "spacemouse_norm_linear_x", "spacemouse_norm_linear_y",
    "spacemouse_norm_linear_z", "spacemouse_norm_angular_x",
    "spacemouse_norm_angular_y", "spacemouse_norm_angular_z",
    "gripper_request_normalized_open_fraction",
)
DEFAULT_MAX_AGE_S = 0.10
DEFAULT_MIN_COVERAGE = 0.90


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            rows.append(item)
    return rows


def _stamp(record: dict[str, Any], key: str, description: str) -> int:
    value = record.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{description}: missing positive {key}")
    return value


def _twist(event: dict[str, Any]) -> list[float]:
    if (event.get("name") != "operator_twist"
            or event.get("topic") != TWIST_TOPIC
            or event.get("message_type") != "geometry_msgs/msg/Twist"):
        raise ValueError("Unexpected SpaceMouse Twist event identity")
    msg = event.get("message")
    if not isinstance(msg, dict):
        raise ValueError("Twist event has no message")
    try:
        values = [
            float(msg[part][axis])
            for part in ("linear", "angular") for axis in ("x", "y", "z")
        ]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Malformed Twist message") from exc
    if not all(math.isfinite(v) and abs(v) <= 1.001 for v in values):
        raise ValueError("Twist inputs must be finite and normalized to [-1,1]")
    return values


def _grip(event: dict[str, Any]) -> float:
    if (event.get("name") != "operator_gripper_request"
            or event.get("topic") != GRIPPER_REQUEST_TOPIC
            or event.get("message_type") != "std_msgs/msg/Float32"):
        raise ValueError("Unexpected gripper-request event identity")
    try:
        value = float(event["message"]["data"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Malformed gripper request") from exc
    if not math.isfinite(value) or value not in (0.0, 1.0):
        raise ValueError("Gripper request must be binary normalized 0.0 or 1.0")
    return value


def _observation_stamp(frame: dict[str, Any]) -> tuple[int, float]:
    # Both camera receipt times are on the workstation's monotonic clock.
    # Use the later camera receipt to avoid associating a command before
    # both images were received. Exposure timing is still UNVERIFIED.
    cameras = frame.get("source_and_camera_metadata", {}).get("cameras", {})
    if not isinstance(cameras, dict):
        raise ValueError("Missing camera receipt metadata")
    stamps = [
        _stamp(cameras.get(role, {}), "received_monotonic_ns", role + " camera")
        for role in ("base", "wrist")
    ]
    skew_s = abs(stamps[0] - stamps[1]) / 1e9
    if skew_s > 0.10:
        raise ValueError(f"Camera receipt skew {skew_s:.4f}s > 0.10s")
    return max(stamps), skew_s


def _mean_held_twist(
    start: int,
    end: int,
    stamps: list[int],
    vectors: list[list[float]],
    max_age_ns: int,
) -> tuple[list[float] | None, float]:
    """ZOH time-weighted mean of REQUESTS; missing/stale intervals not filled."""
    if end <= start:
        raise ValueError("Invalid time window")
    cursor = start
    index = bisect_right(stamps, start) - 1
    integral = np.zeros(6, dtype=np.float64)
    covered = 0
    while cursor < end:
        next_stamp = stamps[index + 1] if index + 1 < len(stamps) else end
        segment_end = min(end, next_stamp)
        if segment_end < cursor:
            raise ValueError("Nonmonotonic command timeline")
        if index >= 0:
            valid_end = min(segment_end, stamps[index] + max_age_ns)
            if valid_end > cursor:
                duration = valid_end - cursor
                integral += np.asarray(vectors[index]) * duration
                covered += duration
        cursor = segment_end
        if cursor < end:
            index += 1
    coverage = covered / (end - start)
    if covered == 0:
        return None, 0.0
    # Divide by known interval, not full interval: never silently fill gaps.
    return (integral / covered).tolist(), coverage


def align_episode(
    ep: Path, *, max_age_s: float = DEFAULT_MAX_AGE_S,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ep = Path(ep)
    if not (0 < max_age_s <= 0.5 and 0 < min_coverage <= 1):
        raise ValueError("Invalid alignment thresholds")
    manifest_path = ep / "manifest.json"
    commands_path = ep / "operator_commands.jsonl"
    manifest = _json(manifest_path)
    if manifest.get("schema") != RAW_SCHEMA or manifest.get("status") != "raw_recorded":
        raise ValueError(f"{ep.name}: raw capture is not complete")
    if manifest.get("operator_events_dropped") != 0:
        raise ValueError(f"{ep.name}: dropped operator requests")
    if manifest.get("executed_arm_commands_available") is True:
        raise ValueError(f"{ep.name}: unexpected controller-accepted action declaration")
    frames = _jsonl(ep / "states.jsonl")
    events = _jsonl(commands_path)
    if len(frames) < 2 or len(frames) != manifest.get("frames_saved"):
        raise ValueError(f"{ep.name}: inconsistent frame count")
    if len(events) != manifest.get("operator_command_events_saved"):
        raise ValueError(f"{ep.name}: inconsistent operator event count")
    frame_stamps = []
    for i, frame in enumerate(frames):
        if frame.get("frame_index") != i or frame.get("training_action") is not None:
            raise ValueError(f"{ep.name}: invalid frame index or unexpected action")
        stamp, camera_skew = _observation_stamp(frame)
        recorded = _stamp(frame, "recorded_monotonic_ns", ep.name)
        if stamp > recorded or (recorded - stamp) > 500_000_000:
            raise ValueError(f"{ep.name}: camera-to-record lag is invalid")
        if frame_stamps and stamp <= frame_stamps[-1]:
            raise ValueError(f"{ep.name}: nonmonotonic camera timestamps")
        # Validate the exact observation contract without changing the raw data.
        candidate_19d_state(frame["measured_state"])
        frame_stamps.append(stamp)
    twists: list[tuple[int, list[float]]] = []
    grippers: list[tuple[int, float]] = []
    for event in events:
        stamp = _stamp(event, "received_monotonic_ns", ep.name + " command")
        if event.get("name") == "operator_twist":
            twists.append((stamp, _twist(event)))
        elif event.get("name") == "operator_gripper_request":
            grippers.append((stamp, _grip(event)))
        else:
            raise ValueError(f"{ep.name}: unexpected command event")
    if not twists:
        raise ValueError(f"{ep.name}: no SpaceMouse Twist requests")
    twists.sort(key=lambda x: x[0])
    grippers.sort(key=lambda x: x[0])
    ts = [x[0] for x in twists]
    vs = [x[1] for x in twists]
    gs = [x[0] for x in grippers]
    rows = []
    count = Counter()
    intervals_s = []
    for i in range(len(frames) - 1):
        start, end = frame_stamps[i:i + 2]
        dt_s = (end - start) / 1e9
        intervals_s.append(dt_s)
        duration_ok = 0.02 <= dt_s <= 0.25
        arm_mean, coverage = (
            _mean_held_twist(start, end, ts, vs, int(max_age_s * 1e9))
            if duration_ok else (None, 0.0)
        )
        # A gripper request is a latched button press. Preserve NULL until
        # the first request occurs; do not derive it from measured knuckle
        # position. End-of-window value includes presses during the window.
        g_idx = bisect_left(gs, end) - 1
        gripper = grippers[g_idx][1] if g_idx >= 0 else None
        complete = (duration_ok and coverage >= min_coverage
                    and arm_mean is not None and gripper is not None)
        if not duration_ok:
            count["invalid_dt"] += 1
        elif coverage < min_coverage:
            count["insufficient_twist_coverage"] += 1
        if gripper is None:
            count["unknown_gripper_request"] += 1
        if complete:
            count["candidate_7d_available"] += 1
        src = frames[i]
        image_refs = src.get("image_files_bgr8", {})
        for role in ("base", "wrist"):
            relative = image_refs.get(role)
            if (not isinstance(relative, str) or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                    or not relative.startswith(f"frames/{role}/")):
                raise ValueError(f"{ep.name}: invalid source image path")
            if not (ep / relative).is_file():
                raise FileNotFoundError(f"{ep.name}: missing source image {relative}")
        rows.append({
            "episode": ep.name,
            "frame_index": i,
            "next_frame_index": i + 1,
            "observation_camera_receipt_ns": start,
            "next_observation_camera_receipt_ns": end,
            "observation_dt_s": dt_s,
            "images_bgr8_npy": image_refs,
            "state19_candidate": candidate_19d_state(src["measured_state"]).tolist(),
            "spacemouse_mean_request_6d": arm_mean,
            "twist_request_time_coverage": round(coverage, 6),
            "gripper_request_open_fraction": gripper,
            "candidate_request_action_7d": (
                [*arm_mean, gripper] if complete else None
            ),
            "candidate_complete": complete,
        })
    review_path = ep / "episode_review.json"
    review = _json(review_path) if review_path.is_file() else {}
    report = {
        "episode": ep.name,
        "raw_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "recorded_requests_sha256": hashlib.sha256(commands_path.read_bytes()).hexdigest(),
        "operator_review_outcome": review.get("operator_outcome"),
        "frames": len(frames),
        "transitions": len(rows),
        "twist_events": len(twists),
        "gripper_request_events": len(grippers),
        "dt_median_s": median(intervals_s),
        "twist_coverage_median": median(
            r["twist_request_time_coverage"] for r in rows
        ),
        "counts": dict(count),
        "controller_actions_verified": False,
        "sft_ready": False,
        "recap_ready": False,
        "physical_execution_approved": False,
    }
    return report, rows


def build_dataset(
    raw_root: Path, output: Path, *, expected_episodes: int = 100,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    raw_root = Path(raw_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output == raw_root or output.is_relative_to(raw_root):
        raise ValueError("Output must not be inside the raw recordings")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    episodes = sorted(x for x in raw_root.glob("episode_*") if x.is_dir())
    if expected_episodes < 1 or len(episodes) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} episodes, found {len(episodes)}"
        )
    # Validate ALL sources before creating any output.
    analyzed = []
    for ep in episodes:
        if ep.is_symlink() or ep.resolve().parent != raw_root:
            raise ValueError("Unexpected symlink/episode path")
        report, rows = align_episode(
            ep, max_age_s=max_age_s, min_coverage=min_coverage
        )
        analyzed.append((report, rows))
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    reports = []
    for report, rows in analyzed:
        name = report["episode"]
        with (output / "episodes" / f"{name}.jsonl").open(
            "x", encoding="utf-8"
        ) as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        reports.append(report)
    result = {
        "schema": SCHEMA,
        "source_root": str(raw_root),
        "output_kind": "external-image-reference JSONL diagnostic dataset",
        "episode_reports": reports,
        "episodes": len(reports),
        "frames": sum(x["frames"] for x in reports),
        "transitions": sum(x["transitions"] for x in reports),
        "candidate_7d_available": sum(
            x["counts"].get("candidate_7d_available", 0) for x in reports
        ),
        "state_contract": CONTRACT_VERSION,
        "action_fields": list(FIELDS),
        "arm_label": "time-weighted mean 6D normalized SpaceMouse request per camera-receipt window",
        "gripper_label": "last recorded normalized button request strictly before window end, or null",
        "max_request_age_s": max_age_s,
        "min_twist_time_coverage": min_coverage,
        "provenance": "workstation ROS topic messages, NOT controller acceptance/execution",
        "not_pose_deltas": True,
        "images_copied": False,
        "camera_exposure_and_controller_latency_calibrated": False,
        "gripper_target_mapping_verified": False,
        "lerobot_exported": False,
        "sft_ready": False,
        "recap_ready": False,
        "physical_execution_approved": False,
        "next_gate": (
            "Audit controller input scaling/frames, update loop timing, "
            "gripper bridge mapping, exposure-to-action latency; "
            "define a distinct normalized-requests OpenPI transform and norm stats. "
            "Do not use existing measured-motion proxy model/norm stats."
        ),
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--max-age-s", type=float, default=DEFAULT_MAX_AGE_S)
    parser.add_argument(
        "--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE
    )
    args = parser.parse_args()
    report = build_dataset(
        args.raw_root, args.output,
        expected_episodes=args.expected_episodes,
        max_age_s=args.max_age_s, min_coverage=args.min_coverage,
    )
    print(
        f"Aligned {report['episodes']} episodes, {report['frames']} frames, "
        f"{report['transitions']} transitions, "
        f"{report['candidate_7d_available']} complete candidate 7D requests"
    )
    print(f"Report: {Path(args.output) / 'report.json'}")
    print(
        "OFFLINE REQUEST DATA ONLY. NOT LeRobot, SFT/RECAP-ready or robot-ready."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
