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

"""Subscription-only ROS 2 state reader; stdout carries JSON records to RLinf."""

import argparse
import json
import os
import signal
import sys
import threading
import time


def main() -> None:
    """Subscribe to existing standard state topics without starting hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-topic", required=True)
    parser.add_argument("--pose-topic", required=True)
    parser.add_argument("--wrench-topic", required=True)
    parser.add_argument("--gripper-topic")
    args = parser.parse_args()
    os.environ["RCUTILS_LOGGING_USE_STDOUT"] = "0"

    import rclpy
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from rosidl_runtime_py.convert import message_to_ordereddict
    from sensor_msgs.msg import JointState

    def emit(record: dict) -> None:
        sys.stdout.write(json.dumps(record, allow_nan=False) + "\n")
        sys.stdout.flush()

    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node(
        "rlinf_franka_observation",
        enable_rosout=False,
        start_parameter_services=False,
    )

    def subscribe(name: str, topic: str, message_type: type) -> None:
        def receive(message: object) -> None:
            try:
                emit(
                    {
                        "event": "state",
                        "name": name,
                        "topic": topic,
                        "message_type": message_type.__module__.split(".")[0]
                        + "/msg/"
                        + message_type.__name__,
                        "units": {
                            "joint_state": {
                                "position": "rad",
                                "velocity": "rad/s",
                                "effort": "N*m",
                            },
                            "tcp_pose": {
                                "position": "m",
                                "orientation": "quaternion_xyzw",
                            },
                            "external_wrench": {"force": "N", "torque": "N*m"},
                            "gripper_state": {"position": "source_joint_definition"},
                        }[name],
                        "message": message_to_ordereddict(message),
                        "received_monotonic_ns": time.monotonic_ns(),
                        "received_unix_ns": time.time_ns(),
                    }
                )
            except (TypeError, ValueError) as error:
                node.get_logger().error(f"Invalid {topic} state: {error}")
                emit({"event": "error", "error": f"Invalid state on {topic}: {error}"})

        node.create_subscription(message_type, topic, receive, qos_profile_sensor_data)

    try:
        subscribe("joint_state", args.joint_topic, JointState)
        subscribe("tcp_pose", args.pose_topic, PoseStamped)
        subscribe("external_wrench", args.wrench_topic, WrenchStamped)
        if args.gripper_topic:
            subscribe("gripper_state", args.gripper_topic, JointState)
        emit(
            {
                "event": "ready",
                "ros_distro": os.environ.get("ROS_DISTRO"),
                "domain_id": os.environ.get("ROS_DOMAIN_ID"),
                "rmw": rclpy.get_rmw_implementation_identifier(),
            }
        )
        while rclpy.ok() and not stopped.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
