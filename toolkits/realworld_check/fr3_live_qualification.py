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

"""Qualify live read-only FR3, Robotiq and dual-camera observations."""

from __future__ import annotations

import argparse
import json
import math
import queue
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from toolkits.realworld_check.fr3_multimodal import assemble_fr3_multimodal
from toolkits.realworld_check.franka_observation import (
    BROADCASTER,
    RosStateSubscriber,
)

SOURCES = ("joint_state", "tcp_pose", "external_wrench", "gripper_state")
CAMERAS = ("base", "wrist")


def summarize_qualification(
    *,
    samples: int,
    valid: int,
    distinct: dict[str, int],
    skews: list[float],
    errors: Counter,
    elapsed_s: float,
    min_valid_fraction: float = 0.95,
    min_distinct_fraction: float = 0.60,
) -> dict:
    """Apply explicit pass criteria to observed, not inferred, samples."""
    if samples <= 0 or elapsed_s <= 0:
        raise ValueError("samples and elapsed_s must be positive")
    if not 0 < min_valid_fraction <= 1 or not 0 < min_distinct_fraction <= 1:
        raise ValueError("Fractions must be between zero and one")
    required = (*SOURCES, *CAMERAS)
    unique = {name: int(distinct.get(name, 0)) for name in required}
    checks = {
        "valid_fraction": valid / samples >= min_valid_fraction,
        "distinct_sources": all(
            unique[name] / samples >= min_distinct_fraction for name in required
        ),
        "camera_skew_samples": len(skews) == valid,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "samples": samples,
        "valid_samples": valid,
        "valid_fraction": valid / samples,
        "distinct_samples": unique,
        "distinct_fraction": {
            name: unique[name] / samples for name in required
        },
        "max_camera_receipt_skew_s": max(skews) if skews else None,
        "p95_camera_receipt_skew_s": float(np.percentile(skews, 95)) if skews else None,
        "elapsed_s": elapsed_s,
        "loop_frequency_hz": samples / elapsed_s,
        "failure_reasons": dict(errors),
        "limits": {
            "min_valid_fraction": min_valid_fraction,
            "min_distinct_fraction": min_distinct_fraction,
        },
    }


def read_camera_pair(cameras: dict, *, max_age_s: float, max_skew_s: float) -> dict:
    """Return fresh BGR camera records with bounded local receipt skew."""
    deadline = time.monotonic() + 2.0
    records = {}

    def read(name: str, age: float) -> None:
        camera = cameras[name]
        stamp, frame = camera.get_frame_with_timestamp(
            timeout=max(0.0, deadline - time.monotonic()), max_age=age
        )
        width, height = camera.camera_info.resolution
        if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"{name} camera expected uint8 {(height, width, 3)}; "
                f"got {frame.shape}, {frame.dtype}"
            )
        records[name] = {
            "frame": frame,
            "serial_number": camera.camera_info.serial_number,
            "encoding": "bgr8",
            "received_monotonic_ns": round(stamp * 1e9),
            "timestamp_kind": "host_receipt_after_driver_read",
        }

    for role in CAMERAS:
        read(role, max_age_s)
    while True:
        older = min(records, key=lambda name: records[name]["received_monotonic_ns"])
        newer = max(records, key=lambda name: records[name]["received_monotonic_ns"])
        skew_s = (records[newer]["received_monotonic_ns"] - records[older]["received_monotonic_ns"]) / 1e9
        if skew_s <= max_skew_s or time.monotonic() >= deadline:
            break
        read(older, 0.0)

    now_ns = time.monotonic_ns()
    for role in CAMERAS:
        age_s = (now_ns - records[role]["received_monotonic_ns"]) / 1e9
        if not 0 <= age_s <= max_age_s:
            raise ValueError(f"{role} camera stale: {age_s:.3f}s")
    return records


def main() -> int:
    """Sample existing controller state and local cameras; never send commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-serial", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--joint-topic", default=BROADCASTER + "/measured_joint_states")
    parser.add_argument("--pose-topic", default=BROADCASTER + "/current_pose")
    parser.add_argument("--wrench-topic", default=BROADCASTER + "/external_wrench_in_base_frame")
    parser.add_argument("--gripper-topic", default="/joint_states")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--fps", type=int, default=15, help="Qualification sample rate")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--base-width", type=int, default=640)
    parser.add_argument("--base-height", type=int, default=480)
    parser.add_argument("--wrist-width", type=int, default=848)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--max-age-s", type=float, default=0.5)
    parser.add_argument("--max-skew-s", type=float, default=0.1)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 20 or args.fps < 1 or args.camera_fps < 1:
        parser.error("--samples must be >=20, and FPS values must be >=1")
    if min(args.base_width, args.base_height, args.wrist_width, args.wrist_height) < 1:
        parser.error("Camera image dimensions must be positive")
    if not args.base_serial or not args.wrist_serial or args.base_serial == args.wrist_serial:
        parser.error("base and wrist must be different physical camera serials")
    if any(not math.isfinite(x) or x <= 0 for x in (args.max_age_s, args.max_skew_s)):
        parser.error("age and skew limits must be positive and finite")

    # Heavy / hardware-specific dependencies are loaded only for a live run.
    from rlinf.robotics.parts.cameras import Camera, CameraInfo

    readers = {
        "arm": RosStateSubscriber(
            args.ros_python,
            domain_id=70,
            joint_topic=args.joint_topic,
            pose_topic=args.pose_topic,
            wrench_topic=args.wrench_topic,
        ),
        "gripper": RosStateSubscriber(
            args.ros_python,
            domain_id=71,
            joint_topic="/fr3_m3b_unused_arm_joint",
            pose_topic="/fr3_m3b_unused_pose",
            wrench_topic="/fr3_m3b_unused_wrench",
            gripper_topic=args.gripper_topic,
        ),
    }
    cameras = {
        "base": Camera.of(CameraInfo(
            name="base", serial_number=args.base_serial,
            resolution=(args.base_width, args.base_height), fps=args.camera_fps
        )),
        "wrist": Camera.of(CameraInfo(
            name="wrist", serial_number=args.wrist_serial,
            resolution=(args.wrist_width, args.wrist_height), fps=args.camera_fps
        )),
    }
    # No SpaceMouse, gripper serial, robot action or FR3 control APIs are opened.
    distinct = Counter()
    previous_stamps: dict[str, int] = {}
    errors = Counter()
    skews = []
    valid = 0
    elapsed_s = 0.0

    with ExitStack() as stack:
        for reader in readers.values():
            reader.start()
            stack.callback(reader.close)
        for camera in cameras.values():
            camera.connect()
            stack.callback(camera.disconnect)

        # Allow the ROS readers to gather their first real messages.
        time.sleep(1.0)
        started = time.monotonic()
        for i in range(args.samples):
            tick = time.monotonic()
            try:
                frames = read_camera_pair(
                    cameras, max_age_s=args.max_age_s, max_skew_s=args.max_skew_s
                )
                arm = readers["arm"].snapshot(args.max_age_s)
                grip = readers["gripper"].snapshot(args.max_age_s)
                result = assemble_fr3_multimodal(
                    arm, grip, frames, max_camera_receipt_skew_s=args.max_skew_s
                )
                valid += 1
                skews.append(result["metadata"]["camera_receipt_skew_s"])
                for name in SOURCES:
                    source = (
                        grip["states"][name]["source"] if name == "gripper_state"
                        else arm["states"][name]["source"]
                    )
                    stamp = source["received_monotonic_ns"]
                    if stamp != previous_stamps.get(name):
                        distinct[name] += 1
                        previous_stamps[name] = stamp
                for role in CAMERAS:
                    stamp = frames[role]["received_monotonic_ns"]
                    if stamp != previous_stamps.get(role):
                        distinct[role] += 1
                        previous_stamps[role] = stamp
            except (ValueError, RuntimeError, TimeoutError, OSError, queue.Empty) as exc:
                errors[f"{type(exc).__name__}: {exc}"] += 1
            time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - tick)))
            if (i + 1) % 15 == 0:
                print(f"sample {i+1}/{args.samples}: {valid} valid", flush=True)
        elapsed_s = time.monotonic() - started

    summary = summarize_qualification(
        samples=args.samples, valid=valid, distinct=dict(distinct), skews=skews,
        errors=errors, elapsed_s=elapsed_s
    )
    summary.update({
        "milestone": "3B_live_observation",
        "hardware_motion": False,
        "domain_ids": {"fr3": 70, "robotiq": 71},
        "camera_serials": {
            "base": args.base_serial, "wrist": args.wrist_serial
        },
        "camera_profiles": {
            name: {
                "width": camera.camera_info.resolution[0],
                "height": camera.camera_info.resolution[1],
                "fps": camera.camera_info.fps,
            }
            for name, camera in cameras.items()
        },
        "topics": {
            "joint_state": args.joint_topic,
            "tcp_pose": args.pose_topic,
            "external_wrench": args.wrench_topic,
            "gripper_state": args.gripper_topic,
        },
        "policy_ready": False,
        "clock_exposure_sync_verified": False,
    })
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
