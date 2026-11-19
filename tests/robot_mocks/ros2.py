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

No ROS graph or hardware is opened. Only whitelisted read-only service clients
and state subscriptions are exposed; command writers and action clients fail.
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
    if topic.endswith("/robot_state"):
        return NS(
            header=header,
            robot_mode=2,
            current_errors=NS(
                joint_reflex=MODE == "fault",
                get_fields_and_field_types=lambda: {"joint_reflex": "boolean"},
            ),
        )
    if topic.endswith("/status"):
        return NS(
            header=header,
            object_detection=3,
            motor_current=0.0,
            gripper_fault=255
            if MODE == "unknown_fault"
            else 9
            if MODE == "fault"
            else 0,
            gripper_fault_severity=255
            if MODE == "unknown_fault"
            else 3
            if MODE == "fault"
            else 0,
        )
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
        self.profile = "fr3" if "fr3" in name else "robotiq"

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
        if topic.endswith("/robot_state"):
            ros_type = "franka_msgs/msg/FrankaRobotState"
        if topic.endswith("/status"):
            ros_type = "robotiq_msgs/msg/GripperStatus"
        if "target_" in topic:
            return []
        return [self.endpoint(ros_type, "state_broadcaster")]

    @staticmethod
    def endpoint(kind, name):
        return NS(
            topic_type=kind,
            node_name=name,
            node_namespace="/",
            qos_profile=NS(
                reliability=NS(name="BEST_EFFORT"),
                durability=NS(name="VOLATILE"),
                depth=5,
            ),
        )

    def get_subscriptions_info_by_topic(self, topic):
        if "target_cartesian" in topic:
            return [
                self.endpoint(
                    "geometry_msgs/msg/Twist", "joint_impedance_ik_controller"
                )
            ]
        return []

    def get_node_names_and_namespaces(self):
        if MODE == "graph_overflow":
            return [("extra_node_" + str(i), "/") for i in range(129)]
        names = (
            ["controller_manager", "joint_impedance_ik_controller"]
            if self.profile == "fr3"
            else [
                "controller_manager",
                "robotiq_gripper_controller",
                "robotiq_output_domain71",
            ]
        )
        if MODE == "ownership_conflict" and self.profile == "robotiq":
            names.append("rogue_client")
        if MODE == "duplicate_manager":
            names.append("controller_manager")
        return [(name, "/") for name in names]

    def create_client(self, kind, name):
        if MODE == "diagnostic_crash":
            os._exit(17)
        if MODE == "diagnostic_hang":
            time.sleep(30)
        allowed = {
            "/controller_manager/list_controllers",
            "/controller_manager/list_hardware_components",
            "/controller_manager/list_hardware_interfaces",
            "/controller_manager/get_parameters",
            "/joint_impedance_ik_controller/get_parameters",
            "/robotiq_gripper_controller/get_parameters",
        }
        assert name in allowed, f"Unexpected service: {name}"
        return NS(
            service_is_ready=lambda: MODE != "service_timeout",
            call_async=lambda request: NS(
                done=lambda: True, result=lambda: self.response(name, request)
            ),
        )

    def response(self, name, request):
        arm = self.profile == "fr3"
        claims = (
            [f"fr3_joint{i}/effort" for i in range(1, 8)]
            if arm
            else ["robotiq_85_left_knuckle_joint/position"]
        )
        commands = [
            NS(
                name=x,
                is_claimed=MODE != "claim_conflict",
                is_available=MODE != "claim_conflict",
            )
            for x in claims
        ]
        if name.endswith("list_controllers"):
            return NS(
                controller=[
                    NS(
                        name="joint_impedance_ik_controller"
                        if arm
                        else "robotiq_gripper_controller",
                        type="franka_arm_controllers/JointImpedanceIKController"
                        if arm
                        else "robotiq_controllers/GripperCommandController",
                        state="active",
                        claimed_interfaces=claims,
                    )
                ]
            )
        if name.endswith("list_hardware_components"):
            return NS(
                component=[
                    NS(
                        name="FrankaHardwareInterface"
                        if arm
                        else "RobotiqGripperHardwareInterface",
                        state=NS(label="active", id=3),
                        command_interfaces=commands,
                    )
                ]
            )
        if name.endswith("list_hardware_interfaces"):
            return NS(command_interfaces=commands)
        params = {
            "update_rate": 1000 if arm else 500,
            "use_sim_time": False,
            "ik_update_rate": 50.0,
            "command_timeout": 0.25,
            "arm_mounting_orientation": [0.0, 0.0, 0.0],
            "arm_id": "fr3",
            "load_gripper": "false",
            "k_gains": [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0],
            "d_gains": [30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0],
            "gripper_profile": "2f_85",
            "gripper_closed_position": 0.7929,
            "max_effort": 50.0,
            "allow_stalling": True,
            "use_object_status": True,
            "object_status_timeout": 10.0,
            "goal_tolerance": 0.02,
            "stall_timeout": 0.05,
            "action_monitor_rate": 100.0,
        }
        if MODE == "config_drift":
            params["command_timeout"] = 0.5

        def encode(value):
            kind = {bool: 1, int: 2, float: 3, str: 4, list: 8}[type(value)]
            return NS(
                type=kind,
                **{
                    {
                        1: "bool_value",
                        2: "integer_value",
                        3: "double_value",
                        4: "string_value",
                        8: "double_array_value",
                    }[kind]: value
                },
            )

        return NS(values=[encode(params[x]) for x in request.names])

    def destroy_client(self, client):
        pass

    def destroy_subscription(self, subscription):
        self.callbacks = {
            topic: cb for topic, cb in self.callbacks.items() if cb is not subscription
        }

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


def action_endpoints(node, name, namespace, *, client):
    if (
        client
        and name
        not in ("robotiq_output_domain71", "rogue_client", "robotiq_gripper_controller")
    ) or (not client and name != "robotiq_gripper_controller"):
        return []
    endpoints = (
        ("gripper_cmd",) if client else ("gripper_cmd", "normalized_gripper_cmd")
    )
    return [
        ("/robotiq_gripper_controller/" + x, ["control_msgs/action/GripperCommand"])
        for x in endpoints
    ]


def get_message(name):
    if MODE == "missing_health_type" and (
        "FrankaRobotState" in name or "GripperStatus" in name
    ):
        raise ImportError("Custom message package unavailable")
    return name


if __name__ == "rclpy":
    module("rclpy.context", Context=Context)
    module("rclpy.node", Node=Node)
    module("rclpy.executors", SingleThreadedExecutor=SingleThreadedExecutor)
    module(
        "rclpy.action",
        get_action_server_names_and_types_by_node=lambda *a: action_endpoints(
            *a, client=False
        ),
        get_action_client_names_and_types_by_node=lambda *a: action_endpoints(
            *a, client=True
        ),
    )
    module(
        "rclpy.qos",
        QoSProfile=lambda **kw: NS(**{"history": "keep_last", **kw}),
        HistoryPolicy=NS(KEEP_LAST="keep_last"),
        ReliabilityPolicy=NS(BEST_EFFORT="best_effort"),
        DurabilityPolicy=NS(VOLATILE="volatile"),
    )
    module("rosidl_runtime_py")
    module(
        "rosidl_runtime_py.utilities",
        get_message=get_message,
        get_service=lambda name: NS(Request=lambda: NS()),
    )
