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

"""Minimal subscriber SDK fake loaded as rclpy in the observation subprocess.

No ROS graph or hardware is opened. The fake has no publisher, client, service,
action, or hardware-control API; accidental use of any of those fails.
"""

import os
import sys
import time
from types import ModuleType
from types import SimpleNamespace as NS

MODE = os.environ.get("RLINF_TEST_ROS2_MODE", "normal")
FR3_NAMES = [f"fr3_joint{i}" for i in (1, 3, 6, 7, 2, 4, 5)]


def state_message(topic, tick=0, joint_name=None):
    """Return standard-message-shaped test data with deliberately shuffled joints."""
    header = NS(stamp=NS(sec=123, nanosec=tick), frame_id="base_link")
    if "joint_states" in topic:
        names = FR3_NAMES if topic == "/franka/joint_states" else [joint_name]
        values = (
            [float(name.removeprefix("fr3_joint")) for name in names]
            if names == FR3_NAMES
            else [0.25 if "normalized" in joint_name else 0.1]
        )
        if MODE == "malformed":
            values = [float("nan")] * len(names)
        return NS(
            header=header,
            name=names,
            position=values,
            velocity=[] if MODE == "partial" else [v / 10 for v in values],
            effort=[v / 100 for v in values],
        )
    if "current_pose" in topic:
        header.frame_id = "base"
        return NS(
            header=header,
            pose=NS(
                position=NS(x=0.1, y=0.2, z=0.3),
                orientation=NS(x=0.0, y=0.0, z=0.6, w=0.8),
            ),
        )
    header.frame_id = "base" if "base_frame" in topic else "stiffness"
    return NS(
        header=header, wrench=NS(force=NS(x=1, y=2, z=3), torque=NS(x=4, y=5, z=6))
    )


class Context:
    def ok(self):
        return True

    def try_shutdown(self):
        pass


def init(*, args, context, domain_id):
    assert str(domain_id) == os.environ["ROS_DOMAIN_ID"]
    assert args == []
    if MODE == "startup_error":
        raise RuntimeError("test ROS initialization failed")


class Node:
    def __init__(
        self,
        name,
        *,
        context,
        enable_rosout,
        start_parameter_services,
        use_global_arguments,
    ):
        assert not (enable_rosout or start_parameter_services or use_global_arguments)
        self.created = time.monotonic()
        self.callbacks = {}
        self.tick = 0

    def get_publishers_info_by_topic(self, topic):
        if MODE == "hang":
            time.sleep(30)
        elapsed = time.monotonic() - self.created
        if (
            MODE == "missing"
            or (MODE == "delayed" and elapsed < 0.6)
            or (MODE == "disconnect" and elapsed > 0.4)
        ):
            return []
        ros_type = (
            "sensor_msgs/msg/JointState"
            if "joint_states" in topic
            else (
                "geometry_msgs/msg/PoseStamped"
                if "current_pose" in topic
                else "geometry_msgs/msg/WrenchStamped"
            )
        )
        if MODE == "wrong_type":
            ros_type = "std_msgs/msg/String"
        return [NS(topic_type=ros_type)]

    def create_subscription(self, msg_type, topic, callback, qos):
        assert qos.depth == 5
        assert (qos.history, qos.reliability, qos.durability) == (
            "keep_last",
            "best_effort",
            "volatile",
        )
        assert "desired" not in topic and "target" not in topic
        self.callbacks[topic] = callback
        return callback

    def destroy_node(self):
        pass


class SingleThreadedExecutor:
    def __init__(self, *, context):
        pass

    def add_node(self, node):
        self.node = node

    def spin_once(self, *, timeout_sec):
        time.sleep(min(timeout_sec, 0.001))
        if MODE == "stale" and self.node.tick > 5:
            return
        self.node.tick += 1
        for topic, callback in self.node.callbacks.items():
            if topic == "/joint_states":
                for name in (
                    "robotiq_85_left_knuckle_joint",
                    "robotiq_normalized_joint",
                    "unrelated_joint",
                ):
                    callback(state_message(topic, self.node.tick, name))
            else:
                callback(state_message(topic, self.node.tick))

    def shutdown(self, *, timeout_sec):
        pass


def module(name, **members):
    value = ModuleType(name)
    value.__dict__.update(members)
    sys.modules[name] = value


if __name__ == "rclpy":
    module("rclpy.context", Context=Context)
    module("rclpy.node", Node=Node)
    module("rclpy.executors", SingleThreadedExecutor=SingleThreadedExecutor)
    module(
        "rclpy.qos",
        QoSProfile=lambda **kw: NS(**kw),
        HistoryPolicy=NS(KEEP_LAST="keep_last"),
        ReliabilityPolicy=NS(BEST_EFFORT="best_effort"),
        DurabilityPolicy=NS(VOLATILE="volatile"),
    )
    module("rosidl_runtime_py")
    module("rosidl_runtime_py.utilities", get_message=lambda name: name)
