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

"""Monitor a read-only Franka observation without constructing a robot or env."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rlinf.robotics.parts.base import Observation


STATE_NAMES = ("joint_state", "tcp_pose", "external_wrench", "gripper_state")
BROADCASTER = "/franka_robot_state_broadcaster"


def state_snapshot(
    records: dict[str, dict], now_ns: int, max_age_s: float, running: bool
) -> dict[str, Any]:
    """Return fresh source records and explicit availability for each state."""
    result = {}
    for name in STATE_NAMES:
        record = records.get(name)
        age = (
            None if record is None else (now_ns - record["received_monotonic_ns"]) / 1e9
        )
        valid = running and age is not None and 0 <= age <= max_age_s
        result[name] = {
            "status": "available" if valid else "unavailable",
            "reason": None
            if valid
            else (
                "subscriber_stopped"
                if not running
                else "missing"
                if age is None
                else "stale"
            ),
            "age_s": age,
            "source": copy.deepcopy(record) if valid else None,
        }
    return result


class RosStateSubscriber:
    """Own a ROS-Python subprocess that subscribes to existing state topics."""

    def __init__(
        self,
        python: str,
        *,
        joint_topic: str,
        pose_topic: str,
        wrench_topic: str,
        gripper_topic: str | None = None,
    ) -> None:
        self.command = [
            python,
            str(Path(__file__).with_name("franka_ros2_observer.py")),
            "--joint-topic",
            joint_topic,
            "--pose-topic",
            pose_topic,
            "--wrench-topic",
            wrench_topic,
        ]
        if gripper_topic:
            self.command.extend(["--gripper-topic", gripper_topic])
        self.process = None
        self.thread = None
        self.records: dict[str, dict] = {}
        self.metadata: dict = {}
        self.error: str | None = None
        self.lock = threading.Lock()
        self.ready = threading.Event()

    def start(self) -> None:
        """Start the subscription process using ROS_DOMAIN_ID=70."""
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("ROS subscriber is already running.")
        self.ready.clear()
        self.records.clear()
        self.metadata.clear()
        self.error = None
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "ROS_DOMAIN_ID": "70",
                "RCUTILS_LOGGING_USE_STDOUT": "0",
            },
        )
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.ready.wait(0.1):
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.close()
                raise RuntimeError(
                    "ROS subscriber did not become ready; source the ROS setup "
                    "and select its system Python with --ros-python."
                )
        if self.error:
            self.close()
            raise RuntimeError(self.error)

    def _receive(self) -> None:
        try:
            for line in self.process.stdout:
                record = json.loads(line)
                with self.lock:
                    if record["event"] == "ready":
                        self.metadata = record
                        self.ready.set()
                    elif record["event"] == "error":
                        raise RuntimeError(record["error"])
                    elif record["event"] == "state" and record["name"] in STATE_NAMES:
                        self.records[record["name"]] = record
                    else:
                        raise ValueError(
                            f"Unexpected ROS subscription record: {record}"
                        )
        except (ValueError, KeyError, RuntimeError) as error:
            with self.lock:
                self.error = str(error)
            self.ready.set()

    def snapshot(self, max_age_s: float) -> dict[str, Any]:
        """Invalidate cached state when receipt is stale or the reader stops."""
        with self.lock:
            running = (
                self.process.poll() is None
                and self.thread.is_alive()
                and not self.error
            )
            return {
                "states": state_snapshot(
                    self.records.copy(), time.monotonic_ns(), max_age_s, bool(running)
                ),
                "transport": {
                    **self.metadata,
                    "running": bool(running),
                    "error": self.error,
                },
            }

    def close(self) -> None:
        """Stop only the subprocess owned by this observer."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            if self.thread is not None:
                self.thread.join(timeout=1)
            self.process.stdout.close()


class FrankaObservation:
    """Combine local RLinf sensors with received ROS state in a raw observation.

    Camera timestamps are host receipts after driver reads. ROS message headers
    retain their source clocks; they are not comparable to host monotonic time.
    """

    def __init__(
        self,
        *,
        base_serial: str,
        wrist_serial: str,
        spacemouse_path: str,
        ros_python: str = "/usr/bin/python3",
        joint_topic: str = BROADCASTER + "/measured_joint_states",
        pose_topic: str = BROADCASTER + "/current_pose",
        wrench_topic: str = BROADCASTER + "/external_wrench_in_base_frame",
        gripper_topic: str | None = None,
        fps: int = 15,
        max_age_s: float = 0.5,
        max_camera_skew_s: float = 0.1,
    ) -> None:
        from filelock import FileLock

        from rlinf.robotics.parts.cameras import Camera, CameraInfo
        from rlinf.robotics.parts.teleop.spacemouse import SpaceMouse

        if not base_serial or not wrist_serial or base_serial == wrist_serial:
            raise ValueError("Expected two distinct, explicit camera serial numbers.")
        if fps <= 0 or any(
            not math.isfinite(value) or value <= 0
            for value in (max_age_s, max_camera_skew_s)
        ):
            raise ValueError(
                "fps, max_age_s and max_camera_skew_s must be positive and finite."
            )
        self.cameras = {
            name: Camera.of(
                CameraInfo(
                    name=name,
                    serial_number=serial,
                    resolution=(640, 480),
                    fps=fps,
                )
            )
            for name, serial in (("base", base_serial), ("wrist", wrist_serial))
        }
        self.spacemouse_path = str(Path(spacemouse_path).resolve())
        self.spacemouse = SpaceMouse(device_path=self.spacemouse_path)
        self.claim = FileLock(
            f"/tmp/rlinf-hid-{Path(self.spacemouse_path).name}.lock", timeout=0
        )
        self.ros = RosStateSubscriber(
            ros_python,
            joint_topic=joint_topic,
            pose_topic=pose_topic,
            wrench_topic=wrench_topic,
            gripper_topic=gripper_topic,
        )
        self.max_age_s = max_age_s
        self.max_camera_skew_s = max_camera_skew_s
        self.stack: ExitStack | None = None

    def connect(self) -> None:
        """Open sensors only; roll back all owned resources on partial failure."""
        if self.stack is not None:
            return
        stack = ExitStack()
        try:
            stack.enter_context(self.claim)
            if not Path(self.spacemouse_path).exists() or shutil.which("fuser") is None:
                raise RuntimeError(
                    "Expected an existing SpaceMouse HID path and fuser."
                )
            owners = subprocess.run(
                ["fuser", self.spacemouse_path],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if owners.returncode != 1:
                raise RuntimeError(
                    f"Cannot establish exclusive SpaceMouse access at {self.spacemouse_path}: "
                    f"{owners.stdout.strip()} {owners.stderr.strip()}"
                )
            self.ros.start()
            stack.callback(self.ros.close)
            for camera in self.cameras.values():
                camera.connect()
                stack.callback(camera.disconnect)
            self.spacemouse.connect()
            stack.callback(self.spacemouse.disconnect)
        except BaseException:
            stack.close()
            raise
        self.stack = stack

    def get_observation(self) -> Observation:
        """Return both real images, raw states, timing and separate operator intent."""
        import numpy as np

        if self.stack is None:
            raise RuntimeError("Call connect() before reading observations.")
        cameras = {}

        def read_camera(name: str, max_age: float) -> None:
            camera = self.cameras[name]
            stamp, frame = camera.get_frame_with_timestamp(
                timeout=max(0, deadline - time.monotonic()), max_age=max_age
            )
            if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                raise ValueError(
                    f"Expected {name} HWC uint8 (480,640,3), got {frame.shape}, {frame.dtype}."
                )
            cameras[name] = {
                "frame": frame,
                "serial_number": camera.camera_info.serial_number,
                "encoding": "bgr8",
                "received_monotonic_ns": round(stamp * 1e9),
                "timestamp_kind": "host_receipt_after_driver_read",
            }

        deadline = time.monotonic() + 2
        for name in self.cameras:
            read_camera(name, self.max_age_s)
        while True:
            older = min(
                cameras, key=lambda name: cameras[name]["received_monotonic_ns"]
            )
            newer = max(
                cameras, key=lambda name: cameras[name]["received_monotonic_ns"]
            )
            skew_s = (
                cameras[newer]["received_monotonic_ns"]
                - cameras[older]["received_monotonic_ns"]
            ) / 1e9
            if skew_s <= self.max_camera_skew_s or time.monotonic() >= deadline:
                break
            read_camera(older, 0)
        now_ns = time.monotonic_ns()
        ages = [
            (now_ns - value["received_monotonic_ns"]) / 1e9
            for value in cameras.values()
        ]
        if (
            any(age < 0 or age > self.max_age_s for age in ages)
            or skew_s > self.max_camera_skew_s
        ):
            raise RuntimeError(
                f"Camera pair is stale or misaligned: ages={ages}, skew_s={skew_s}."
            )
        intent = self.spacemouse.get_observation()
        if not np.all(np.isfinite(intent["twist"])):
            raise ValueError("Expected finite SpaceMouse axes.")
        return {
            "robot": self.ros.snapshot(self.max_age_s),
            "cameras": cameras,
            "operator_intent": {
                **intent,
                "read_completed_monotonic_ns": time.monotonic_ns(),
                "axis_order": ["-y", "x", "z", "-roll", "-pitch", "-yaw"],
                "units": "normalized_device_input",
                "operator_orientation": "RLinf_default",
                "source_timestamp": None,
            },
            "timing": {
                "assembled_monotonic_ns": time.monotonic_ns(),
                "assembled_unix_ns": time.time_ns(),
                "camera_receipt_skew_s": skew_s,
                "max_age_s": self.max_age_s,
                "max_camera_skew_s": self.max_camera_skew_s,
                "synchronization": "bounded_host_receipt_skew_not_exposure_sync",
            },
        }

    def disconnect(self) -> None:
        """Close owned subscriptions and sensors; repeated calls are harmless."""
        if self.stack is not None:
            stack, self.stack = self.stack, None
            stack.close()

    def __enter__(self) -> FrankaObservation:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()


def main() -> None:
    """Run a bounded metadata monitor without saving images or demonstrations."""
    from rlinf.utils.logging import get_logger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-serial", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--spacemouse-path", required=True)
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--joint-topic", default=BROADCASTER + "/measured_joint_states")
    parser.add_argument("--pose-topic", default=BROADCASTER + "/current_pose")
    parser.add_argument(
        "--wrench-topic", default=BROADCASTER + "/external_wrench_in_base_frame"
    )
    parser.add_argument("--gripper-topic")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2.")
    options = vars(args).copy()
    samples = options.pop("samples")
    output = options.pop("summary_json")
    stamps = {"base": [], "wrist": []}
    skews = []
    with FrankaObservation(**options) as monitor:
        started = time.monotonic()
        for _ in range(samples):
            observation = monitor.get_observation()
            for name in stamps:
                stamp = observation["cameras"][name]["received_monotonic_ns"]
                if not stamps[name] or stamps[name][-1] != stamp:
                    stamps[name].append(stamp)
            skews.append(observation["timing"]["camera_receipt_skew_s"])
            time.sleep(0.1)
        summary = {
            "observations": samples,
            "elapsed_s": time.monotonic() - started,
            "cameras": {
                name: {
                    "serial_number": camera["serial_number"],
                    "shape": list(camera["frame"].shape),
                    "dtype": str(camera["frame"].dtype),
                    "encoding": camera["encoding"],
                    "configured_fps": monitor.cameras[name].camera_info.fps,
                    "distinct_frames": len(stamps[name]),
                    "observed_unique_frame_rate_hz": (
                        (len(stamps[name]) - 1)
                        * 1e9
                        / (stamps[name][-1] - stamps[name][0])
                        if len(stamps[name]) > 1
                        else None
                    ),
                }
                for name, camera in observation["cameras"].items()
            },
            "max_camera_receipt_skew_s": max(skews),
            "operator_intent": {
                "twist": observation["operator_intent"]["twist"].tolist(),
                "buttons": observation["operator_intent"]["buttons"].tolist(),
                "source_timestamp": None,
            },
            "robot": observation["robot"],
            "timing": observation["timing"],
        }
    get_logger().info("Read-only observation verification: %s", json.dumps(summary))
    if output:
        output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
