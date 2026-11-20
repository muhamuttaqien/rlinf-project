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

"""Unregistered synthetic parts for the offline FR3 environment only."""

import time
from typing import Any

import numpy as np

from .base import Features, Observation
from .cameras.base import Camera
from .ros2_observation import FR3ROS2Observation, RobotiqROS2Observation
from .teleop.base import TeleopPart
from .transports.ros2.action_contract import LOCAL_CLOCK_DOMAIN
from .transports.ros2.synthetic import SyntheticROS2ObservationTransport


class SyntheticFR3Observation(FR3ROS2Observation):
    """Reuse FR3 state mapping with an in-process fixture, never a ROS worker."""

    def __init__(self) -> None:
        super().__init__(70)
        self._transport = SyntheticROS2ObservationTransport("fr3", 70)


class SyntheticRobotiqObservation(RobotiqROS2Observation):
    """Reuse raw-angle/closure mapping without claiming calibrated jaw width."""

    def __init__(self) -> None:
        super().__init__(71)
        self._transport = SyntheticROS2ObservationTransport("robotiq", 71)


class SyntheticCamera(Camera):
    """Generate RGB HWC fixture arrays without selecting a camera backend."""

    def __init__(self, width: int, height: int, sensor: str) -> None:
        self.width, self.height, self.sensor = width, height, sensor

    def _open(self) -> object:
        return object()

    def _release(self, device: Any) -> None:
        pass

    @property
    def observation_features(self) -> Features:
        """Describe a synthetic RGB uint8 image and its provenance."""
        return {
            "image": {"shape": (self.height, self.width, 3), "dtype": "uint8"},
            "metadata": {},
        }

    def get_observation(self) -> Observation:
        """Read a fixture frame, independent of any physical camera."""
        if not self.is_connected:
            raise ConnectionError("Expected a connected synthetic camera")
        return {
            "image": np.zeros((self.height, self.width, 3), dtype=np.uint8),
            "metadata": {
                "synthetic": True,
                "source": "fixture:" + self.sensor,
                "receipt_monotonic_ns": time.monotonic_ns(),
                "clock_domain": LOCAL_CLOCK_DOMAIN,
                "source_timestamp": None,
                "encoding": "RGB",
                "layout": "HWC",
            },
        }


class SyntheticSpaceMouse(TeleopPart):
    """Expose raw fixture input only; no operator-to-controller action mapping."""

    def _open(self) -> object:
        return object()

    def _release(self, device: Any) -> None:
        pass

    @property
    def observation_features(self) -> Features:
        """Describe six raw axes and two buttons without a robot action contract."""
        return {"axes": {"shape": (6,)}, "buttons": {"shape": (2,)}, "metadata": {}}

    def get_observation(self) -> Observation:
        """Read a synthetic neutral input; it is never converted to an action."""
        if not self.is_connected:
            raise ConnectionError("Expected a connected synthetic SpaceMouse")
        return {
            "axes": np.zeros(6, dtype=np.float32),
            "buttons": np.zeros(2, dtype=bool),
            "metadata": {
                "synthetic": True,
                "source": "fixture:spacemouse",
                "receipt_monotonic_ns": time.monotonic_ns(),
                "clock_domain": LOCAL_CLOCK_DOMAIN,
                "source_timestamp": None,
                "action_mapping": None,
            },
        }
