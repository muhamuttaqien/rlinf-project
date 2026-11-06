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

"""Record raw FR3 and Robotiq observations, images and operator command events.

This module is read-only. It never produces robot actions, LeRobot targets,
7D policy actions or synthetic success labels.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.realworld_check.fr3_live_qualification import read_camera_pair
from toolkits.realworld_check.fr3_multimodal import assemble_fr3_multimodal
from toolkits.realworld_check.franka_observation import BROADCASTER, RosStateSubscriber

RAW_SCHEMA = "fr3_robotiq_raw_capture_v1"
TWIST_TOPIC = "/franka_controller/target_cartesian_velocity_percent"
GRIPPER_REQUEST_TOPIC = "/gripper_client/target_gripper_width_percent"


class RawCaptureWriter:
    """Persist one unlabeled raw episode; never overwrite existing data."""

    def __init__(
        self,
        root: Path,
        *,
        task: str,
        requested_samples: int,
        sample_rate_hz: int,
        camera_profiles: dict[str, dict[str, Any]],
        topic_names: dict[str, str],
    ) -> None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("An explicit nonempty task description is required")
        if requested_samples < 1 or sample_rate_hz < 1:
            raise ValueError("Requested samples and sampling rate must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        for role in ("base", "wrist"):
            (self.root / "frames" / role).mkdir(parents=True)
        self._states = (self.root / "states.jsonl").open("x", encoding="utf-8")
        self._commands = (self.root / "operator_commands.jsonl").open("x", encoding="utf-8")
        self.count = 0
        self.command_count = 0
        self.dropped_commands = 0
        self._closed = False
        self.manifest: dict[str, Any] = {
            "schema": RAW_SCHEMA,
            "status": "recording",
            "task": task,
            "label": "unlabeled_raw_observation",
            "realworld_episode_success": None,
            "is_lerobot_dataset": False,
            "policy_ready": False,
            "action_targets_7d_available": False,
            "executed_arm_commands_available": False,
            "cross_host_exposure_sync_verified": False,
            "requested_samples": requested_samples,
            "sample_rate_hz_requested": sample_rate_hz,
            "camera_profiles": camera_profiles,
            "topics": topic_names,
            "frames_saved": 0,
            "operator_command_events_saved": 0,
            "operator_events_dropped": 0,
            "started_unix_ns": time.time_ns(),
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def write_sample(self, observation: dict[str, Any]) -> None:
        """Save two lossless BGR frames and a validated measured-state record."""
        if self._closed:
            raise RuntimeError("Capture writer has already closed")
        if observation["metadata"]["policy_ready"] is not False:
            raise ValueError("Expected raw, not policy-ready, observation")
        index = self.count
        frame_files = {}
        for role in ("base", "wrist"):
            relpath = f"frames/{role}/{index:06d}.npy"
            with (self.root / relpath).open("xb") as stream:
                np.save(stream, observation["frames"][role], allow_pickle=False)
            frame_files[role] = relpath
        record = {
            "frame_index": index,
            "recorded_monotonic_ns": time.monotonic_ns(),
            "recorded_unix_ns": time.time_ns(),
            "measured_state": {
                key: np.asarray(value).tolist()
                for key, value in observation["state"].items()
            },
            "source_and_camera_metadata": observation["metadata"],
            "image_files_bgr8": frame_files,
            "operator_command_labels": None,
            "training_action": None,
        }
        self._states.write(json.dumps(record, allow_nan=False) + "\n")
        self._states.flush()
        self.count += 1

    def write_commands(self, events: list[dict], dropped: int = 0) -> None:
        """Keep raw ROS topic events separate from measured robot states."""
        if self._closed:
            raise RuntimeError("Capture writer has already closed")
        if dropped < 0:
            raise ValueError("Dropped event count cannot be negative")
        self.dropped_commands += dropped
        for event in events:
            if event.get("name") not in (
                "operator_twist", "operator_gripper_request"
            ):
                raise ValueError("Unexpected operator command topic")
            self._commands.write(json.dumps(event, allow_nan=False) + "\n")
            self.command_count += 1
        self._commands.flush()

    def finish(self, *, status: str, failures: dict[str, int] | None = None) -> None:
        """Finalize as raw-only or incomplete; never claim a demonstration."""
        if self._closed:
            return
        if status not in ("raw_recorded", "incomplete", "interrupted"):
            raise ValueError("Unknown raw-record status")
        if self.dropped_commands and status == "raw_recorded":
            status = "incomplete"
        self._states.close()
        self._commands.close()
        self.manifest.update({
            "status": status,
            "frames_saved": self.count,
            "operator_command_events_saved": self.command_count,
            "operator_events_dropped": self.dropped_commands,
            "failures": failures or {},
            "completed_unix_ns": time.time_ns(),
        })
        self._write_manifest()
        self._closed = True

    def __enter__(self) -> RawCaptureWriter:
        return self

    def __exit__(self, exc_type: type | None, *_: Any) -> None:
        if not self._closed:
            self.finish(status="interrupted" if exc_type else "incomplete")


def main() -> int:
    """Capture a bounded read-only pilot using the previously qualified sensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--base-serial", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--base-width", type=int, default=640)
    parser.add_argument("--base-height", type=int, default=480)
    parser.add_argument("--wrist-width", type=int, default=848)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
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
    if min(
        args.base_width, args.base_height, args.wrist_width, args.wrist_height,
        args.camera_fps, args.fps, args.samples
    ) < 1:
        parser.error("Camera sizes, FPS and samples must be positive")
    if args.base_serial == args.wrist_serial or not args.base_serial or not args.wrist_serial:
        parser.error("Camera serial numbers must be distinct and nonempty")
    if any(not math.isfinite(x) or x <= 0 for x in (args.max_age_s, args.max_skew_s)):
        parser.error("Age/skew thresholds must be positive finite values")

    # Import hardware drivers only in a live run; unit tests use the writer alone.
    from rlinf.robotics.parts.cameras import Camera, CameraInfo

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
    arm = RosStateSubscriber(
        args.ros_python, domain_id=70,
        joint_topic=args.joint_topic, pose_topic=args.pose_topic,
        wrench_topic=args.wrench_topic,
        twist_topic=args.twist_topic,
        gripper_request_topic=args.gripper_request_topic,
    )
    gripper = RosStateSubscriber(
        args.ros_python, domain_id=71,
        joint_topic="/fr3_raw_unused_joint",
        pose_topic="/fr3_raw_unused_pose",
        wrench_topic="/fr3_raw_unused_wrench",
        gripper_topic=args.gripper_topic,
    )
    camera_profiles = {
        name: {
            "serial_number": camera.camera_info.serial_number,
            "resolution": list(camera.camera_info.resolution),
            "fps": camera.camera_info.fps,
        }
        for name, camera in cameras.items()
    }
    topic_names = {
        "joint_state": args.joint_topic,
        "tcp_pose": args.pose_topic,
        "external_wrench": args.wrench_topic,
        "gripper_state": args.gripper_topic,
        "operator_twist": args.twist_topic,
        "operator_gripper_request": args.gripper_request_topic,
    }
    failures = Counter()
    with RawCaptureWriter(
        args.output, task=args.task, requested_samples=args.samples,
        sample_rate_hz=args.fps, camera_profiles=camera_profiles,
        topic_names=topic_names
    ) as writer:
        with ExitStack() as stack:
            for reader in (arm, gripper):
                reader.start()
                stack.callback(reader.close)
            for camera in cameras.values():
                camera.connect()
                stack.callback(camera.disconnect)
            time.sleep(1.0)
            # Discard events received before the capture window begins.
            arm.drain_events()
            started = time.monotonic()
            for i in range(args.samples):
                tick = time.monotonic()
                try:
                    frames = read_camera_pair(
                        cameras, max_age_s=args.max_age_s, max_skew_s=args.max_skew_s
                    )
                    observation = assemble_fr3_multimodal(
                        arm.snapshot(args.max_age_s),
                        gripper.snapshot(args.max_age_s),
                        frames,
                        max_camera_receipt_skew_s=args.max_skew_s,
                    )
                    writer.write_sample(observation)
                except (ValueError, RuntimeError, TimeoutError, OSError, queue.Empty) as exc:
                    failures[f"{type(exc).__name__}: {exc}"] += 1
                events, dropped = arm.drain_events()
                writer.write_commands(events, dropped)
                time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - tick)))
                if (i + 1) % 10 == 0:
                    print(f"sample {i+1}/{args.samples}: {writer.count} raw frames saved", flush=True)
            events, dropped = arm.drain_events()
            writer.write_commands(events, dropped)
            elapsed = time.monotonic() - started
        status = (
            "raw_recorded"
            if writer.count >= math.ceil(0.95 * args.samples)
            and writer.dropped_commands == 0
            else "incomplete"
        )
        writer.finish(status=status, failures=dict(failures))
        print(json.dumps({
            "status": status,
            "raw_frames": writer.count,
            "requested_samples": args.samples,
            "operator_events": writer.command_count,
            "dropped_operator_events": writer.dropped_commands,
            "elapsed_s": elapsed,
            "failures": dict(failures),
            "directory": str(args.output),
            "lerobot_ready": False,
            "training_action_labels": False,
        }, indent=2))
    return 0 if status == "raw_recorded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
