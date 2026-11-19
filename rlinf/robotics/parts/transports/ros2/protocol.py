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

"""Bounded local IPC and standard ROS state decoding (stdlib only)."""

import json
import math
import socket
from typing import Any

VERSION = 1
MAX_PACKET = 65536
FR3_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))
RAW_KNUCKLE = "robotiq_85_left_knuckle_joint"
NORMALIZED_JOINT = "robotiq_normalized_joint"
PREFIX = "/franka_robot_state_broadcaster/"
# stream: (topic, ROS type, selected joint names)
PROFILES = {
    "fr3": {
        "joints": ("/franka/joint_states", "sensor_msgs/msg/JointState", FR3_JOINTS),
        "pose": (PREFIX + "current_pose", "geometry_msgs/msg/PoseStamped", ()),
        "wrench_base": (
            PREFIX + "external_wrench_in_base_frame",
            "geometry_msgs/msg/WrenchStamped",
            (),
        ),
        "wrench_stiffness": (
            PREFIX + "external_wrench_in_stiffness_frame",
            "geometry_msgs/msg/WrenchStamped",
            (),
        ),
    },
    "robotiq": {
        "knuckle": ("/joint_states", "sensor_msgs/msg/JointState", (RAW_KNUCKLE,)),
        "normalized": (
            "/joint_states",
            "sensor_msgs/msg/JointState",
            (NORMALIZED_JOINT,),
        ),
    },
}


def send_packet(channel: socket.socket, packet: dict[str, Any]) -> None:
    """Send one versioned JSON packet over a Unix SOCK_SEQPACKET socket."""
    data = json.dumps({"version": VERSION, **packet}, allow_nan=False).encode()
    if len(data) > MAX_PACKET:
        raise ValueError(f"IPC packet exceeds {MAX_PACKET} bytes: {len(data)}")
    if channel.send(data) != len(data):
        raise ConnectionError("Expected a complete IPC packet; send was truncated")


def receive_packet(channel: socket.socket) -> dict[str, Any]:
    """Receive a bounded packet; reject EOF, truncation, and version mismatch."""
    data = channel.recv(MAX_PACKET + 1)
    if not data:
        raise ConnectionError("ROS observation worker closed its IPC socket")
    if len(data) > MAX_PACKET:
        raise ValueError(f"IPC packet exceeds {MAX_PACKET} bytes")
    result = json.loads(data)
    if not isinstance(result, dict) or result.get("version") != VERSION:
        raise ValueError(f"Expected IPC protocol version {VERSION}")
    return result


def finite_vector(values: Any, size: int) -> list[float]:
    """Validate one finite vector with an exact dimension."""
    result = [float(value) for value in values]
    if len(result) != size or not all(math.isfinite(v) for v in result):
        raise ValueError(f"Expected {size} finite values; got {result}")
    return result


def decode_message(msg: Any, ros_type: str, joints: tuple[str, ...]) -> dict | None:
    """Decode measured standard state, returning None for unrelated joints.

    Joint velocity and effort may be absent under the JointState contract.
    Gripper velocity is deliberately excluded: the driver reports a fixed zero.
    """
    if ros_type == "sensor_msgs/msg/JointState":
        names = list(msg.name)
        if not set(joints).intersection(names):
            return None
        if len(names) != len(set(names)) or not set(joints).issubset(names):
            raise ValueError(
                f"Expected unique joint names including {joints}; got {names}"
            )
        indices = [names.index(name) for name in joints]
        data = {"names": list(joints)}
        fields = (
            ("position", "velocity", "effort")
            if joints == FR3_JOINTS
            else ("position",)
        )
        for field in fields:
            values = getattr(msg, field)
            vector = finite_vector(values, len(names)) if len(values) else None
            data[field] = None if vector is None else [vector[i] for i in indices]
        if joints == (NORMALIZED_JOINT,) and data["position"] is not None:
            if not 0 <= data["position"][0] <= 1:
                raise ValueError("Expected normalized Robotiq closure in [0, 1]")
    elif ros_type == "geometry_msgs/msg/PoseStamped":
        p, q = msg.pose.position, msg.pose.orientation
        pose = finite_vector([p.x, p.y, p.z, q.x, q.y, q.z, q.w], 7)
        if not math.isclose(sum(v * v for v in pose[3:]), 1.0, abs_tol=1e-3):
            raise ValueError("Expected a unit quaternion in x,y,z,w order")
        data = {"tcp_pose": pose}
    elif ros_type == "geometry_msgs/msg/WrenchStamped":
        f, t = msg.wrench.force, msg.wrench.torque
        data = {
            "force": finite_vector([f.x, f.y, f.z], 3),
            "torque": finite_vector([t.x, t.y, t.z], 3),
        }
    else:
        raise ValueError(f"Unsupported observation message type: {ros_type}")
    stamp = msg.header.stamp
    if not isinstance(stamp.sec, int) or not 0 <= stamp.nanosec < 1_000_000_000:
        raise ValueError("Expected a ROS timestamp with nanosec in [0, 1000000000)")
    if not isinstance(msg.header.frame_id, str):
        raise ValueError("Expected a string frame_id")
    return {
        "data": data,
        "header": {
            "sec": stamp.sec,
            "nanosec": stamp.nanosec,
            "frame_id": msg.header.frame_id,
        },
    }
