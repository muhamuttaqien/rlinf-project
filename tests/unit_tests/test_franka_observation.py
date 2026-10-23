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

"""Tests for raw observation availability and preservation of ROS state."""

import pytest

from toolkits.realworld_check.franka_observation import (
    FrankaObservation,
    state_snapshot,
)


def joint_record():
    return {
        "event": "state",
        "name": "joint_state",
        "topic": "/franka_robot_state_broadcaster/measured_joint_states",
        "received_monotonic_ns": 1_000_000_000,
        "received_unix_ns": 1_792_000_000_000_000_000,
        "message": {
            "header": {"stamp": {"sec": 123, "nanosec": 456}, "frame_id": "fr3_link0"},
            "name": ["fr3_joint1", "fr3_joint2"],
            "position": [0.1, -0.2],
            "velocity": [0.3, 0.4],
            "effort": [1.0, 2.0],
        },
    }


def test_raw_state_preserves_source_clock_frame_and_joint_velocity():
    record = joint_record()
    result = state_snapshot({"joint_state": record}, 1_100_000_000, 0.5, True)
    assert result["joint_state"]["status"] == "available"
    assert result["joint_state"]["age_s"] == pytest.approx(0.1)
    assert result["joint_state"]["source"] == record
    result["joint_state"]["source"]["message"]["velocity"][0] = 99
    assert record["message"]["velocity"] == [0.3, 0.4]
    assert result["tcp_pose"]["source"] is None
    assert result["tcp_pose"]["reason"] == "missing"


@pytest.mark.parametrize(
    "now_ns,running,reason",
    [
        (2_000_000_000, True, "stale"),
        (900_000_000, True, "stale"),
        (1_100_000_000, False, "subscriber_stopped"),
    ],
)
def test_invalid_state_is_unavailable_without_fabricated_values(
    now_ns, running, reason
):
    result = state_snapshot({"joint_state": joint_record()}, now_ns, 0.5, running)
    assert result["joint_state"]["source"] is None
    assert result["joint_state"]["reason"] == reason
    assert result["gripper_state"]["source"] is None


def test_camera_roles_cannot_refer_to_the_same_device():
    with pytest.raises(ValueError, match="two distinct"):
        FrankaObservation(
            base_serial="same",
            wrist_serial="same",
            spacemouse_path="/dev/hidraw4",
        )
