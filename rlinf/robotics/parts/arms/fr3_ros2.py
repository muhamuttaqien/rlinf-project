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

"""Native request arm attached to an externally owned ROS 2 FR3 controller."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from rlinf.robotics.parts.arms.base import Arm


@Arm.register("fr3_ros2_request")
class FR3ROS2RequestArm(Arm):
    """Own one system-Python ROS peer; commands are normalized requests."""

    def __init__(
        self,
        session_dir: str,
        ros_python: str = "/usr/bin/python3",
        ros_setup: str = "/opt/ros/jazzy/setup.bash",
        ros_overlay: str = "/tmp/fr3_calibration/jazzy_msgs/install/local_setup.bash",
    ) -> None:
        self.session_dir = Path(session_dir)
        self.ros_python = ros_python
        self.ros_setup = ros_setup
        self.ros_overlay = ros_overlay
        self._rpc_lock = threading.Lock()
        self._stderr = None

    @property
    def observation_features(self) -> dict[str, Any]:
        """Expose the checkpoint's measured 19D request state."""
        return {"state": {"shape": (19,), "dtype": "float32"}}

    @property
    def action_features(self) -> dict[str, Any]:
        """Expose six normalized arm requests and one gripper prediction."""
        return {"request": {"shape": (7,), "dtype": "float32"}}

    def _open(self) -> Any:
        from rlinf.robotics.parts.transports.ros2 import fr3_runtime

        script = Path(fr3_runtime.__file__).with_name("fr3_node.py")
        self._stderr = (self.session_dir / "ros.stderr").open("x")
        process = subprocess.Popen(
            [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                'source "$1" && source "$2" && exec "$3" -B "$4" --session "$5" --log "$6"',
                "fr3-ros-peer",
                self.ros_setup,
                self.ros_overlay,
                self.ros_python,
                str(script),
                str(self.session_dir),
                str(self.session_dir / "ros_evidence.jsonl"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "RCUTILS_LOGGING_USE_STDOUT": "0",
            },
        )
        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    self._exchange(process, {"op": "observe"})
                    break
                except RuntimeError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            self._exchange(process, {"op": "reserve"})
            while True:
                try:
                    self._exchange(process, {"op": "observe"})
                    return process
                except RuntimeError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        except BaseException:
            self._release(process)
            raise

    def _exchange(self, process: Any, command: dict[str, Any]) -> dict[str, Any]:
        """Bound each RPC; the child owns refresh and expiry while we wait."""
        with self._rpc_lock:
            if process.poll() is not None:
                raise RuntimeError(
                    f"ROS peer exited; inspect {self.session_dir / 'ros.stderr'}"
                )
            process.stdin.write(json.dumps(command, allow_nan=False) + "\n")
            process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                if not selector.select(timeout=3):
                    (self.session_dir / "STOP").touch()
                    raise RuntimeError("ROS peer RPC timed out; stop latched")
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("ROS peer closed its response stream")
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Invalid ROS peer response"))
            return response

    def get_observation(self) -> dict[str, Any]:
        """Read fresh physical state and source receipt times."""
        return self._exchange(self._device, {"op": "observe"})

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send one expiring request without scaling or frame conversion."""
        from rlinf.robotics.parts.transports.ros2.fr3_runtime import request_vector

        value = request_vector(action["request"])
        return self._exchange(
            self._device,
            {
                "op": "request",
                "value": value,
                "issued": time.monotonic(),
            },
        )

    def stop(self) -> None:
        """Latch stop through the shared local file, even when RPC is busy."""
        (self.session_dir / "STOP").touch()

    def _release(self, device: Any) -> None:
        self.stop()
        if device.poll() is None:
            device.stdin.close()
            try:
                device.wait(timeout=2)
            except subprocess.TimeoutExpired:
                device.terminate()
                try:
                    device.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    device.kill()
                    device.wait(timeout=1)
        if device.stdout is not None:
            device.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
