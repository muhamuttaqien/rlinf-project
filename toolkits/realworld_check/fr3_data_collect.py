# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manually start/stop raw FR3 episodes without sending robot commands.

This captures measured FR3/Robotiq state, two cameras, and existing ROS
SpaceMouse command events. Outcomes are kept in a separate review sidecar;
raw manifests and actions remain unlabeled for the existing preflight.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import select
import shutil
import sys
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from toolkits.realworld_check.fr3_live_qualification import read_camera_pair
from toolkits.realworld_check.fr3_multimodal import assemble_fr3_multimodal
from toolkits.realworld_check.fr3_raw_capture import (
    GRIPPER_REQUEST_TOPIC,
    RAW_SCHEMA,
    TWIST_TOPIC,
    RawCaptureWriter,
)
from toolkits.realworld_check.franka_observation import BROADCASTER, RosStateSubscriber

SESSION_SCHEMA = "fr3_manual_episode_session_v1"
REVIEW_SCHEMA = "fr3_operator_episode_review_v1"
EPISODE_NAME = re.compile(r"episode_(\d{4,})$")

DEFAULT_DATA_ROOT = Path("/data1/Muttaqien")


def warn_if_outside_data_root(output: Path, data_root: Path = DEFAULT_DATA_ROOT) -> bool:
    """Warn about large camera recordings outside the workstation data volume."""
    resolved = output.expanduser().resolve()
    allowed = data_root.expanduser().resolve()
    if resolved == allowed or allowed in resolved.parents:
        return False
    print(
        f"WARNING: Large FR3 recordings are outside {allowed}: {resolved}. "
        "Prefer --output-root /data1/Muttaqien/fr3_peg_100.",
        file=sys.stderr, flush=True,
    )
    return True



def prepare_session(root: Path, settings: dict[str, Any]) -> None:
    """Create a session or resume only with its original capture contract."""
    meta_path = root / "session.json"
    if root.exists():
        if not root.is_dir() or not meta_path.is_file():
            raise ValueError("Existing output root is not a valid episode session")
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved != settings:
            raise ValueError("Existing session has different task, camera or sample settings")
        return
    root.mkdir(parents=True, exist_ok=False)
    meta_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def session_progress(root: Path) -> tuple[int, int]:
    """Return next never-reused index and count operator-reviewed raw successes."""
    next_index, successes = 1, 0
    for path in root.iterdir():
        match = EPISODE_NAME.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        next_index = max(next_index, int(match.group(1)) + 1)
        review_path = path / "episode_review.json"
        if not review_path.is_file():
            continue
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if (review.get("schema") == REVIEW_SCHEMA
                and review.get("operator_outcome") == "success"
                and review.get("capture_status") == "raw_recorded"):
            successes += 1
    return next_index, successes


def discard_aborted_episode(path: Path, session_root: Path) -> None:
    """Delete only the just-recorded, recognized episode directory on abort."""
    if (not EPISODE_NAME.fullmatch(path.name) or path.is_symlink()
            or path.parent.resolve() != session_root.resolve()
            or not path.is_dir()):
        raise ValueError("Refusing to delete an invalid episode path")
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Refusing to delete episode without a manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RAW_SCHEMA:
        raise ValueError("Refusing to delete episode with unexpected manifest schema")
    shutil.rmtree(path)


def capture_episode(
    root: Path, *, task: str, fps: int, max_samples: int,
    camera_profiles: dict[str, dict], topic_names: dict[str, str],
    observation: Callable[[], dict], drain_events: Callable[[], tuple[list[dict], int]],
    stop_requested: Callable[[], bool], sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Collect one variable-duration raw episode with an explicit stop."""
    failures: Counter[str] = Counter()
    stop_reason = "max_duration"
    with RawCaptureWriter(
        root, task=task, requested_samples=max_samples, sample_rate_hz=fps,
        camera_profiles=camera_profiles, topic_names=topic_names,
    ) as writer:
        start_mono = time.monotonic_ns()
        start_unix = time.time_ns()
        try:
            for tick_index in range(max_samples):
                if stop_requested():
                    stop_reason = "operator_stop"
                    break
                tick = time.monotonic()
                try:
                    writer.write_sample(observation())
                except (ValueError, RuntimeError, TimeoutError, OSError, queue.Empty) as exc:
                    failures[f"{type(exc).__name__}: {exc}"] += 1
                events, dropped = drain_events()
                writer.write_commands(events, dropped)
                sleep(max(0.0, 1 / fps - (time.monotonic() - tick)))
                if (tick_index + 1) % 50 == 0:
                    print(f"  samples: {writer.count}", flush=True)
            events, dropped = drain_events()
            writer.write_commands(events, dropped)
        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
        end_mono = time.monotonic_ns()
        end_unix = time.time_ns()
        valid = (stop_reason == "operator_stop" and writer.count > 0
                 and not failures and writer.dropped_commands == 0)
        status = ("interrupted" if stop_reason == "keyboard_interrupt"
                  else "raw_recorded" if valid else "incomplete")
        writer.finish(status=status, failures=dict(failures))
        return {
            "capture_status": status,
            "stop_reason": stop_reason,
            "frames": writer.count,
            "operator_events": writer.command_count,
            "dropped_operator_events": writer.dropped_commands,
            "capture_start_workstation_monotonic_ns": start_mono,
            "capture_end_workstation_monotonic_ns": end_mono,
            "capture_start_workstation_unix_ns": start_unix,
            "capture_end_workstation_unix_ns": end_unix,
            "capture_failures": dict(failures),
        }


def keyboard_stop() -> bool:
    """Consume an Enter line if available, without blocking acquisition."""
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


def prompt_outcome() -> str:
    while True:
        value = input("Outcome [Enter/s]uccess / [f]ailure / [a]borted: ").strip().lower()
        if value in {"", "s", "f", "a"}:
            return {"": "success", "s": "success", "f": "failure", "a": "aborted"}[value]
        print("Please press Enter or enter s, f, or a.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target-successes", type=int, default=100)
    parser.add_argument("--max-duration-s", type=float, default=300.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--base-serial", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--base-width", type=int, default=640)
    parser.add_argument("--base-height", type=int, default=480)
    parser.add_argument("--wrist-width", type=int, default=848)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--controller-log", required=True,
                        help="NUC JSONL path for provenance only; not read or synchronized")
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--joint-topic", default=BROADCASTER + "/measured_joint_states")
    parser.add_argument("--pose-topic", default=BROADCASTER + "/current_pose")
    parser.add_argument("--wrench-topic", default=BROADCASTER + "/external_wrench_in_base_frame")
    parser.add_argument("--gripper-topic", default="/joint_states")
    parser.add_argument("--twist-topic", default=TWIST_TOPIC)
    parser.add_argument("--gripper-request-topic", default=GRIPPER_REQUEST_TOPIC)
    parser.add_argument("--max-age-s", type=float, default=0.5)
    parser.add_argument("--max-skew-s", type=float, default=0.1)
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("Interactive episode capture requires a terminal (TTY)")
    if (not args.task.strip() or not args.controller_log.strip()
            or args.base_serial == args.wrist_serial):
        parser.error("Task, controller log, and distinct camera serials are required")
    if min(args.target_successes, args.fps, args.base_width, args.base_height,
           args.wrist_width, args.wrist_height, args.camera_fps) < 1:
        parser.error("Counts, sampling rates and camera sizes must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in (
            args.max_duration_s, args.max_age_s, args.max_skew_s)):
        parser.error("Duration and age/skew thresholds must be finite and positive")
    max_samples = math.ceil(args.max_duration_s * args.fps)
    topic_names = {
        "joint_state": args.joint_topic, "tcp_pose": args.pose_topic,
        "external_wrench": args.wrench_topic, "gripper_state": args.gripper_topic,
        "operator_twist": args.twist_topic,
        "operator_gripper_request": args.gripper_request_topic,
    }
    profiles = {
        "base": {"serial_number": args.base_serial,
                 "resolution": [args.base_width, args.base_height], "fps": args.camera_fps},
        "wrist": {"serial_number": args.wrist_serial,
                  "resolution": [args.wrist_width, args.wrist_height], "fps": args.camera_fps},
    }
    settings = {
        "schema": SESSION_SCHEMA, "task": args.task, "fps": args.fps,
        "camera_profiles": profiles, "topics": topic_names,
    }
    root = args.output_root.expanduser()
    warn_if_outside_data_root(root)
    prepare_session(root, settings)
    index, successes = session_progress(root)
    print(f"Existing reviewed successes: {successes}/{args.target_successes}")
    if successes >= args.target_successes:
        return 0

    # Keep sensor connections open across attempts; no ROS publishers are created.
    from rlinf.robotics.parts.cameras import Camera, CameraInfo

    cameras = {
        role: Camera.of(CameraInfo(
            name=role, serial_number=profile["serial_number"],
            resolution=tuple(profile["resolution"]), fps=args.camera_fps,
        ))
        for role, profile in profiles.items()
    }
    arm = RosStateSubscriber(
        args.ros_python, domain_id=70, joint_topic=args.joint_topic,
        pose_topic=args.pose_topic, wrench_topic=args.wrench_topic,
        twist_topic=args.twist_topic, gripper_request_topic=args.gripper_request_topic,
    )
    gripper = RosStateSubscriber(
        args.ros_python, domain_id=71,
        joint_topic="/fr3_raw_unused_joint", pose_topic="/fr3_raw_unused_pose",
        wrench_topic="/fr3_raw_unused_wrench", gripper_topic=args.gripper_topic,
    )
    try:
        with ExitStack() as stack:
            for reader in (arm, gripper):
                reader.start()
                stack.callback(reader.close)
            for camera in cameras.values():
                camera.connect()
                stack.callback(camera.disconnect)
            time.sleep(1)
            while successes < args.target_successes:
                answer = input(f"Episode {index:04d}: Enter to START, q to quit: ").strip().lower()
                if answer == "q":
                    break
                if answer:
                    print("Enter or q only.")
                    continue
                arm.drain_events()  # exclude between-episode operator events

                def observe() -> dict:
                    frames = read_camera_pair(
                        cameras, max_age_s=args.max_age_s, max_skew_s=args.max_skew_s,
                    )
                    return assemble_fr3_multimodal(
                        arm.snapshot(args.max_age_s), gripper.snapshot(args.max_age_s),
                        frames, max_camera_receipt_skew_s=args.max_skew_s,
                    )

                print("🔴 RECORDING — press Enter to STOP (Ctrl-C aborts).", flush=True)
                path = root / f"episode_{index:04d}"
                result = capture_episode(
                    path, task=args.task, fps=args.fps, max_samples=max_samples,
                    camera_profiles=profiles, topic_names=topic_names,
                    observation=observe, drain_events=arm.drain_events,
                    stop_requested=keyboard_stop,
                )
                outcome = ("aborted" if result["capture_status"] == "interrupted"
                           else prompt_outcome())
                if outcome == "aborted":
                    discard_aborted_episode(path, root)
                    print(f"Discarded {path}: aborted; successes "
                          f"{successes}/{args.target_successes}")
                    if result["capture_status"] == "interrupted":
                        break
                    continue  # Retry the same index; nothing from this attempt remains.
                review = {
                    "schema": REVIEW_SCHEMA, "episode_index": index,
                    "task": args.task, "operator_outcome": outcome,
                    "operator_reviewed": True,
                    "controller_telemetry_nuc_path": args.controller_log,
                    "controller_time_alignment_verified": False,
                    "metric_training_actions_verified": False,
                    "is_lerobot_dataset": False,
                    **result,
                }
                with (path / "episode_review.json").open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(review, indent=2) + "\n")
                if outcome == "success" and result["capture_status"] == "raw_recorded":
                    successes += 1
                print(f"Saved {path}: {outcome}, {result['capture_status']}, "
                      f"{result['frames']} frames; successes {successes}/{args.target_successes}")
                index += 1
                if result["capture_status"] == "interrupted":
                    break
    except (KeyboardInterrupt, EOFError):
        print("\nSession stopped; existing episodes retained.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
