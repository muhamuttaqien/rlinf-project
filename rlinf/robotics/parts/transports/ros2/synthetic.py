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

"""In-process synthetic state packets using the existing ROS observation decoder."""

import time
from types import SimpleNamespace as NS
from typing import Any

from .observation import ROS2ObservationTransport
from .protocol import FR3_JOINTS, PROFILES, decode_message


class SyntheticROS2ObservationTransport(ROS2ObservationTransport):
    """Generate marked fixtures without importing ROS or starting a worker."""

    def __init__(self, profile: str, domain_id: int) -> None:
        super().__init__(profile, domain_id)
        if domain_id != {"fr3": 70, "robotiq": 71}[profile]:
            raise ValueError("Expected synthetic FR3 domain 70 or Robotiq domain 71")
        self._synthetic_alive = False
        self._sequence = 0

    @property
    def is_alive(self) -> bool:
        """Return only the local fixture lifecycle state."""
        return self._synthetic_alive

    def start(self) -> None:
        """Enable fixture reads without opening any socket or device."""
        self._synthetic_alive = True

    def close(self) -> None:
        """Release the in-memory fixture."""
        self._synthetic_alive = False

    def inspect_status(self, budget: float = 2.0) -> dict[str, Any]:
        """Refuse ROS diagnostics on a synthetic fixture."""
        raise NotImplementedError("Synthetic observations have no ROS graph")

    def snapshot(self) -> dict[str, Any]:
        """Decode fresh synthetic messages, independent of mock command outcomes."""
        if not self.is_alive:
            raise ConnectionError("Expected a connected synthetic observation fixture")
        self._sequence += 1
        sec, nano = divmod(time.time_ns(), 10**9)
        streams = {}
        for key, (topic, kind, joints) in PROFILES[self.profile].items():
            header = NS(stamp=NS(sec=sec, nanosec=nano), frame_id="synthetic/base_link")
            if kind == "sensor_msgs/msg/JointState":
                if self.profile == "fr3":
                    names = list(reversed(FR3_JOINTS))
                    msg = NS(
                        header=header,
                        name=names,
                        position=[0.0] * 7,
                        velocity=[0.0] * 7,
                        effort=[0.0] * 7,
                    )
                else:
                    msg = NS(
                        header=header,
                        name=list(joints),
                        position=[0.1 if key == "knuckle" else 0.25],
                        velocity=[],
                        effort=[],
                    )
            elif kind == "geometry_msgs/msg/PoseStamped":
                header.frame_id = "synthetic/base"
                msg = NS(
                    header=header,
                    pose=NS(
                        position=NS(x=0.0, y=0.0, z=0.0),
                        orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                )
            else:
                header.frame_id = (
                    "synthetic/base" if key == "wrench_base" else "synthetic/fr3_link2"
                )
                msg = NS(
                    header=header,
                    wrench=NS(
                        force=NS(x=0.0, y=0.0, z=0.0), torque=NS(x=0.0, y=0.0, z=0.0)
                    ),
                )
            sample = decode_message(msg, kind, joints)
            sample.update(
                received_monotonic_ns=time.monotonic_ns(),
                synthetic=True,
                source="offline_synthetic_fixture",
            )
            streams[key] = {
                "topic": topic,
                "type": kind,
                "sample": sample,
                "sequence": self._sequence,
                "publishers": 1,
                "error": None,
                "graph_error": None,
                "synthetic": True,
            }
        return {
            "op": "snapshot",
            "domain_id": self.domain_id,
            "profile": self.profile,
            "streams": streams,
            "synthetic": True,
        }
