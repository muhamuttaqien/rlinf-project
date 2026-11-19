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

"""Franka request-policy task using the existing NUC controller and bridge."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rlinf.robotics.parts.arms.fr3_ros2 import FR3ROS2RequestArm
from rlinf.robotics.parts.cameras import Camera, CameraInfo
from rlinf.robotics.robots.franka import FrankaRobot


@dataclass
class FR3RequestConfig:
    """Task observation contract and externally owned controller attachment."""

    session_dir: str
    ros_python: str = "/usr/bin/python3"
    ros_setup: str = "/opt/ros/jazzy/setup.bash"
    ros_overlay: str = "/tmp/fr3_calibration/jazzy_msgs/install/local_setup.bash"
    task_description: str = "Insert the peg into the hole."
    manual_episode_control_only: bool = True
    is_dummy: bool = False


class FR3RequestEnv(gym.Env):
    """Single supervised request episode; reset observes without homing."""

    TELEOP = ()
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Any = None,
        robot_info: Any = None,
        env_idx: int = 0,
    ) -> None:
        self.config = FR3RequestConfig(**override_cfg)
        if env_idx != 0 or self.config.is_dummy:
            raise ValueError(
                "FR3 request evaluation requires one explicitly attached robot"
            )
        self.task_description = self.config.task_description
        self.arm = FR3ROS2RequestArm(
            self.config.session_dir,
            self.config.ros_python,
            self.config.ros_setup,
            self.config.ros_overlay,
        )
        self.base_camera = Camera.backend("realsense")(
            CameraInfo("base", "337122075473", resolution=(640, 480), fps=30)
        )
        self.wrist_camera = Camera.backend("realsense")(
            CameraInfo("wrist", "218622277827", resolution=(848, 480), fps=30)
        )
        self.robot = FrankaRobot(
            arm=self.arm, base=self.base_camera, wrist=self.wrist_camera
        )
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(7,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "request_state": gym.spaces.Box(
                            -np.inf, np.inf, shape=(19,), dtype=np.float32
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        "base": gym.spaces.Box(
                            0, 255, shape=(480, 640, 3), dtype=np.uint8
                        ),
                        "wrist": gym.spaces.Box(
                            0, 255, shape=(480, 848, 3), dtype=np.uint8
                        ),
                    }
                ),
            }
        )
        self._reset_once = False
        self._stopped = False
        self._next_slot = None
        self._first_in_chunk = True
        self._observation_time = -np.inf
        self._chunk_observation_time = -np.inf
        try:
            self.robot.connect()
        except BaseException:
            self.robot.disconnect()
            raise

    def _observe(self) -> dict[str, Any]:
        started = time.monotonic()
        reading = self.arm.get_observation()
        stamps, frames = [reading["received"]], {}
        for name, camera in (("base", self.base_camera), ("wrist", self.wrist_camera)):
            stamp, bgr = camera.get_frame_with_timestamp(timeout=0.25, max_age=0.25)
            stamps.append(stamp)
            frames[name] = np.ascontiguousarray(bgr[..., ::-1])
        self._observation_time = min(stamps)
        if time.monotonic() - self._observation_time > 0.25:
            raise RuntimeError("Camera/ROS observation exceeded 250 ms receipt age")
        self._stopped |= bool(reading["stopped"])
        self._record_timing(
            "observation_acquired",
            started=started,
            received=self._observation_time,
            source_receipts=stamps,
        )
        return {
            "state": {"request_state": np.asarray(reading["state"], dtype=np.float32)},
            "frames": frames,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """Acquire one initial observation; refuse rearming or automatic reset."""
        super().reset(seed=seed)
        if self._reset_once:
            raise RuntimeError(
                "FR3 request evaluation permits one episode per operator confirmation"
            )
        self._reset_once = True
        self._last_obs = self._observe()
        return self._last_obs, {}

    def on_action_chunk_begin(self) -> None:
        """Bind the next horizon to the observation used for policy inference."""
        self._chunk_observation_time = self._observation_time
        self._first_in_chunk = True
        self._next_slot = None
        self._record_timing(
            "action_chunk_received", received=self._chunk_observation_time
        )

    def _record_timing(self, event: str, **values: Any) -> None:
        """Record local monotonic timing without issuing robot commands."""
        path = Path(self.config.session_dir) / "env_timing.jsonl"
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    {"event": event, "monotonic": time.monotonic(), **values},
                    allow_nan=False,
                )
                + "\n"
            )

    def get_hold_action(self, fallback: Any = None) -> np.ndarray:
        """Neutral arm with no binary gripper edge."""
        return np.asarray([0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple:
        """Admit each unchanged request on 100 ms slots within the local deadline."""
        if self._stopped:
            return self._last_obs, 0.0, False, True, {"stop_latched": True}
        try:
            now = time.monotonic()
            if self._first_in_chunk:
                self._record_timing(
                    "action_admission",
                    received=self._chunk_observation_time,
                    observation_age_s=now - self._chunk_observation_time,
                )
                if now - self._chunk_observation_time > 1.0:
                    raise RuntimeError(
                        "Policy inference observation exceeded one second: "
                        f"age={now - self._chunk_observation_time:.6f}s"
                    )
                self._first_in_chunk = False
            if self._next_slot is None:
                self._next_slot = now
            time.sleep(max(0, self._next_slot - now))
            if time.monotonic() - self._next_slot > 0.05:
                raise RuntimeError(
                    "10 Hz policy action missed its slot by more than 50 ms"
                )
            self._record_timing("request_issued", received=self._chunk_observation_time)
            result = self.robot.send_action(
                {"arm": {"request": np.asarray(action).tolist()}}
            )
            self._record_timing("request_acknowledged", **result["arm"])
            self._stopped |= bool(result["arm"]["stopped"])
            self._last_obs = self._observe()
            self._next_slot += 0.1
            time.sleep(max(0, self._next_slot - time.monotonic()))
            deadline = result["arm"].get("deadline")
            self._stopped |= deadline is not None and time.monotonic() >= deadline
            return (
                self._last_obs,
                0.0,
                False,
                self._stopped,
                {"stop_latched": self._stopped},
            )
        except BaseException:
            self._stopped = True
            self.arm.stop()
            raise

    def close(self) -> None:
        """Latch neutral and close only resources owned by this task."""
        self.arm.stop()
        self.robot.disconnect()
