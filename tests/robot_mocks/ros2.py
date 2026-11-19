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

"""ROS 2 vendor boundary for the native FR3 subprocess and task tests."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS


def vector(x=0.0, y=0.0, z=0.0):
    return NS(x=x, y=y, z=z)


class Twist:
    def __init__(self):
        self.linear, self.angular = vector(), vector()


class Float32:
    def __init__(self, data=0.0):
        self.data = data


class Errors:
    fault = False

    def get_fields_and_field_types(self):
        return {"fault": "boolean"}


def message(topic, domain=70):
    stamp = time.monotonic_ns()
    header = NS(stamp=NS(sec=stamp // 10**9, nanosec=stamp % 10**9), frame_id="base")
    if topic.endswith("current_pose"):
        return NS(
            header=header,
            pose=NS(
                position=vector(0.4, 0.0, 0.4),
                orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )
    if topic.endswith("measured_joint_states"):
        return NS(
            header=header,
            name=[f"fr3_joint{i}" for i in range(7, 0, -1)],
            position=[i * 0.1 for i in range(7, 0, -1)],
        )
    if topic.endswith("base_frame"):
        return NS(
            header=header,
            wrench=NS(force=vector(1.0, 2.0, 3.0), torque=vector(3, 4, 0)),
        )
    if topic.endswith("robot_state"):
        return NS(header=header, robot_mode=2, current_errors=Errors())
    if domain == 71:
        return NS(
            header=header, name=["robotiq_85_left_knuckle_joint"], position=[0.7859]
        )
    return None


class Context:
    pass


class Node:
    def __init__(self, name, domain):
        self.name, self.domain = name, domain
        self.subscriptions, self.publishers = [], {}
        self.extra_publishers = 0
        self.subscriber_count = 1
        self.active = True
        self.frozen = False
        self._index = 0

    def create_subscription(self, kind, topic, callback, qos):
        self.subscriptions.append((topic, callback))
        return object()

    def create_publisher(self, kind, topic, qos):
        node = self
        self.publishers[topic] = []

        class Publisher:
            def publish(self, msg):
                value = (
                    [
                        msg.linear.x,
                        msg.linear.y,
                        msg.linear.z,
                        msg.angular.x,
                        msg.angular.y,
                        msg.angular.z,
                    ]
                    if isinstance(msg, Twist)
                    else msg.data
                )
                node.publishers[topic].append(value)
                destination = os.environ.get("FR3_FAKE_WIRE_LOG")
                if destination:
                    with Path(destination).open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "fake_hardware": True,
                                    "topic": topic,
                                    "value": value,
                                    "time": time.monotonic(),
                                }
                            )
                            + "\n"
                        )

        return Publisher()

    def count_publishers(self, topic):
        return int(topic in self.publishers) + self.extra_publishers

    def count_subscribers(self, topic):
        return self.subscriber_count

    def create_client(self, kind, topic):
        node = self

        class Client:
            def service_is_ready(self):
                return True

            def call_async(self, request):
                class Future:
                    def done(self):
                        return True

                    def result(self):
                        return NS(
                            controller=[
                                NS(
                                    name="joint_impedance_ik_controller",
                                    state="active" if node.active else "inactive",
                                    claimed_interfaces=[
                                        f"fr3_joint{i}/effort" for i in range(1, 8)
                                    ],
                                )
                            ]
                        )

                return Future()

        return Client()

    def destroy_node(self):
        pass


class Executor:
    def __init__(self, context):
        self.node = None

    def add_node(self, node):
        self.node = node

    def spin_once(self, timeout_sec=0):
        node = self.node
        if node.frozen or not node.subscriptions:
            return
        topic, callback = node.subscriptions[node._index % len(node.subscriptions)]
        node._index += 1
        msg = message(topic, node.domain)
        if msg is not None:
            callback(msg)

    def shutdown(self, timeout_sec=0):
        pass


def install():
    """Install fake ROS SDK modules before opening the production transport."""

    def module(name, **fields):
        item = ModuleType(name)
        item.__dict__.update(fields)
        sys.modules[name] = item
        return item

    def init(*, context, domain_id):
        context.domain = domain_id

    module(
        "rclpy",
        init=init,
        shutdown=lambda **kw: None,
        create_node=lambda name, context: Node(name, context.domain),
    )
    module("rclpy.context", Context=Context)
    module("rclpy.executors", SingleThreadedExecutor=Executor)
    module("rclpy.qos", qos_profile_sensor_data=object())
    for package in (
        "controller_manager_msgs",
        "franka_msgs",
        "geometry_msgs",
        "sensor_msgs",
        "std_msgs",
        "rcl_interfaces",
    ):
        module(package)
    module(
        "controller_manager_msgs.srv",
        ListControllers=NS(Request=type("Request", (), {})),
    )
    module("franka_msgs.msg", FrankaRobotState=object)
    module("geometry_msgs.msg", PoseStamped=object, WrenchStamped=object, Twist=Twist)
    module("sensor_msgs.msg", JointState=object)
    module("std_msgs.msg", Float32=Float32)
    module("rcl_interfaces.msg", Log=object)


def realsense():
    """Return a two-serial RGB camera SDK honoring each stream's resolution."""
    import numpy as np

    serials = ("337122075473", "218622277827")

    class Config:
        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, stream, width, height, *rest):
            self.shape = (height, width, 3)

        def disable_all_streams(self):
            pass

    class Pipeline:
        def start(self, config):
            self.config = config
            return object()

        def wait_for_frames(self):
            image = np.zeros(self.config.shape, dtype=np.uint8)
            image[..., 0], image[..., 1], image[..., 2] = 11, 22, 33
            return NS(
                get_depth_frame=lambda: None,
                get_color_frame=lambda: NS(
                    is_video_frame=lambda: True, get_data=lambda: image
                ),
            )

        def stop(self):
            pass

    module = ModuleType("pyrealsense2")
    module.pipeline, module.config = Pipeline, Config
    module.context = lambda: NS(
        devices=[NS(get_info=lambda key, s=s: s) for s in serials]
    )
    module.camera_info, module.stream = NS(serial_number=0), NS(color=1, depth=2)
    module.format = NS(bgr8=1, z16=2)
    module.align = lambda stream: NS(process=lambda frames: frames)
    return module
