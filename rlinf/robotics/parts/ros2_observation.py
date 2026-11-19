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

"""Read-only FR3 and Robotiq ROS 2 parts for native robot composition."""

import math
import time
from typing import Any, ClassVar

from .base import Features, Observation, RobotPart
from .transports.ros2.observation import ROS2ObservationTransport
from .transports.ros2.protocol import PROFILES


class ROS2ObservationPart(RobotPart):
    """A subscriber-only part with explicit partial-state and freshness metadata.

    Args:
        domain_id: Explicit ROS domain, independent of the parent environment.
        max_age: Positive maximum local receipt age in seconds.
        ros_python: Python interpreter compatible with the chosen ROS installation.
        ros_setup: ROS setup.bash sourced only in the child process.
        timeout: Startup and individual IPC request timeout in seconds.
    """

    PROFILE: ClassVar[str]
    STATE_FIELDS: ClassVar[tuple[str, ...]]
    UNAVAILABLE_FIELDS: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        domain_id: int,
        *,
        max_age: float = 0.25,
        ros_python: str = "/usr/bin/python3",
        ros_setup: str = "/opt/ros/jazzy/setup.bash",
        timeout: float = 5.0,
    ) -> None:
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError(f"Expected positive finite max_age; got {max_age}")
        self.domain_id, self.max_age = domain_id, max_age
        self._transport = ROS2ObservationTransport(
            self.PROFILE,
            domain_id,
            ros_python=ros_python,
            ros_setup=ros_setup,
            timeout=timeout,
        )

    def _open(self) -> ROS2ObservationTransport:
        self._transport.start()
        return self._transport

    def _release(self, device: ROS2ObservationTransport) -> None:
        device.close()

    @property
    def observation_features(self) -> Features:
        """Describe the measured subset and its ROS metadata."""
        return {"state": {name: {} for name in self.STATE_FIELDS}, "ros2": {}}

    def get_state(self) -> Observation:
        """Return latest samples, including missing/stale/error diagnostics.

        ``complete`` requires every selected stream and field to be fresh and
        valid. Freshness measures callback receipt age on this host, not sensor
        acquisition age. Retained stale samples remain visible with flags.
        """
        error = None
        streams = {}
        alive = self.is_connected and self._transport.is_alive
        if self.is_connected:
            try:
                streams = self._transport.snapshot()["streams"]
            except ConnectionError as exc:
                error, alive = str(exc), False
        else:
            error = "Observation part is disconnected; call connect() first"
        now = time.monotonic_ns()
        missing, stale, invalid = [], [], []
        for key, (topic, ros_type, _) in PROFILES[self.PROFILE].items():
            stream = streams.setdefault(
                key,
                {
                    "topic": topic,
                    "type": ros_type,
                    "sample": None,
                    "sequence": 0,
                    "publishers": 0,
                    "error": None,
                },
            )
            sample = stream["sample"]
            age = (
                None
                if sample is None
                else (now - sample["received_monotonic_ns"]) / 1e9
            )
            stream["age_s"] = age
            stream["fresh"] = age is not None and 0 <= age <= self.max_age
            stream["connected"] = alive and stream["publishers"] > 0
            if sample is None:
                missing.append(key)
            elif not stream["fresh"]:
                stale.append(key)
            if stream["error"] or stream.get("graph_error") or not stream["connected"]:
                invalid.append(key)
        state = self._state_from(streams)
        missing_fields = [key for key, value in state.items() if value is None]
        return {
            "state": state,
            "ros2": {
                "domain_id": self.domain_id,
                "bridge_alive": alive,
                "complete": alive
                and not (missing or stale or invalid or missing_fields),
                "missing_streams": missing,
                "stale_streams": stale,
                "invalid_streams": invalid,
                "missing_fields": missing_fields,
                "unavailable_fields": list(self.UNAVAILABLE_FIELDS),
                "max_age_s": self.max_age,
                "error": error,
                "streams": streams,
            },
        }

    def get_observation(self) -> Observation:
        """Read through the native per-owner snapshot contract."""
        return self.read_state()

    def inspect_status(self, budget: float = 2.0) -> dict[str, Any]:
        """Query fixed passive diagnostics through the owned observation worker."""
        if not self.is_connected:
            raise ConnectionError("Expected a connected observation part")
        return self._transport.inspect_status(budget)

    def wait_for_observation(self, timeout: float = 5.0) -> Observation:
        """Wait for a complete selected observation or raise with diagnostics."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError(f"Expected nonnegative finite timeout; got {timeout}")
        deadline = time.monotonic() + timeout
        while True:
            observation = self.get_state()
            status = observation["ros2"]
            if status["complete"]:
                return observation
            if not status["bridge_alive"]:
                raise ConnectionError(
                    status["error"] or "ROS observation worker exited"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Expected complete ROS observations within {timeout}s; got {status}"
                )
            time.sleep(min(0.02, remaining))

    def _state_from(self, streams: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _data(streams: dict[str, Any], key: str) -> dict[str, Any]:
        sample = streams[key]["sample"]
        return {} if sample is None else sample["data"]


class FR3ROS2Observation(ROS2ObservationPart):
    """Measured FR3 joints, pose and external wrench on domain 70.

    Pose uses metres followed by quaternion x,y,z,w. Wrench values retain their
    separate base/stiffness frame headers in ``ros2.streams``.
    """

    PROFILE = "fr3"
    STATE_FIELDS = (
        "arm_joint_position",
        "arm_joint_velocity",
        "arm_joint_effort",
        "tcp_pose",
        "tcp_force",
        "tcp_torque",
        "base_force",
        "base_torque",
    )
    UNAVAILABLE_FIELDS = ("tcp_vel", "arm_jacobian")

    def __init__(self, domain_id: int = 70, **settings: Any) -> None:
        super().__init__(domain_id, **settings)

    def _state_from(self, streams: dict[str, Any]) -> dict[str, Any]:
        joints = self._data(streams, "joints")
        pose = self._data(streams, "pose")
        base = self._data(streams, "wrench_base")
        stiffness = self._data(streams, "wrench_stiffness")
        return {
            "arm_joint_position": joints.get("position"),
            "arm_joint_velocity": joints.get("velocity"),
            "arm_joint_effort": joints.get("effort"),
            "tcp_pose": pose.get("tcp_pose"),
            "tcp_force": stiffness.get("force"),
            "tcp_torque": stiffness.get("torque"),
            "base_force": base.get("force"),
            "base_torque": base.get("torque"),
        }


class RobotiqROS2Observation(ROS2ObservationPart):
    """Robotiq 2F-85 raw knuckle angle and normalized closure on domain 71.

    Raw position is in radians; normalized closure is 0=open, 1=closed.
    Neither is a calibrated opening width in metres.
    """

    PROFILE = "robotiq"
    STATE_FIELDS = ("knuckle_position_rad", "normalized_closure")
    UNAVAILABLE_FIELDS = (
        "opening_width_m",
        "measured_velocity",
        "grip_force",
        "typed_status",
    )

    def __init__(self, domain_id: int = 71, **settings: Any) -> None:
        super().__init__(domain_id, **settings)

    def _state_from(self, streams: dict[str, Any]) -> dict[str, Any]:
        state = {}
        for key, name in (
            ("knuckle", "knuckle_position_rad"),
            ("normalized", "normalized_closure"),
        ):
            position = self._data(streams, key).get("position")
            state[name] = None if position is None else position[0]
        return state
