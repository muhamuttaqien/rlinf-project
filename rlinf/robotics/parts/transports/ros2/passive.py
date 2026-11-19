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

"""Fixed read-only ROS diagnostics, imported only by the ROS-compatible worker."""

import math
import time
from typing import Any

STATE_TOPICS = {
    "fr3": (
        "/franka/joint_states",
        "/franka_robot_state_broadcaster/current_pose",
        "/franka_robot_state_broadcaster/external_wrench_in_base_frame",
        "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
    ),
    "robotiq": ("/joint_states",),
}
COMMAND_TOPIC = "/franka_controller/target_cartesian_velocity_percent"
BRIDGE_TOPIC = "/gripper_client/target_gripper_width_percent"
GRIPPER_ACTIONS = (
    "/robotiq_gripper_controller/gripper_cmd",
    "/robotiq_gripper_controller/normalized_gripper_cmd",
)
HEALTH = {
    "fr3": (
        "/franka_robot_state_broadcaster/robot_state",
        "franka_msgs/msg/FrankaRobotState",
    ),
    "robotiq": (
        "/robotiq_gripper_status_broadcaster/status",
        "robotiq_msgs/msg/GripperStatus",
    ),
}
PARAMETERS = {
    "fr3": {
        "/controller_manager": ("update_rate", "use_sim_time"),
        "/joint_impedance_ik_controller": (
            "ik_update_rate",
            "command_timeout",
            "arm_mounting_orientation",
            "arm_id",
            "load_gripper",
            "k_gains",
            "d_gains",
            "use_sim_time",
        ),
    },
    "robotiq": {
        "/controller_manager": ("update_rate", "use_sim_time"),
        "/robotiq_gripper_controller": (
            "gripper_profile",
            "gripper_closed_position",
            "max_effort",
            "allow_stalling",
            "use_object_status",
            "object_status_timeout",
            "goal_tolerance",
            "stall_timeout",
            "action_monitor_rate",
            "use_sim_time",
        ),
    },
}


def bounded(items: Any, limit: int) -> list[Any]:
    """Reject excessive evidence instead of silently truncating it."""
    result = list(items)
    if len(result) > limit:
        raise ValueError(
            f"Expected at most {limit} diagnostic entries; got {len(result)}"
        )
    return result


def interfaces(items: Any) -> list[dict[str, Any]]:
    """Copy reported resource availability and claims with a size bound."""
    return [
        {"name": x.name, "claimed": x.is_claimed, "available": x.is_available}
        for x in bounded(items, 128)
    ]


def parameter(value: Any) -> Any:
    """Decode supported parameter values; unavailable values remain None."""
    fields = {
        1: "bool_value",
        2: "integer_value",
        3: "double_value",
        4: "string_value",
        6: "bool_array_value",
        7: "integer_array_value",
        8: "double_array_value",
        9: "string_array_value",
    }
    if value.type not in fields:
        return None
    result = getattr(value, fields[value.type])
    return bounded(result, 32) if value.type >= 6 else result


def endpoints(items: Any) -> list[dict[str, Any]]:
    """Copy discovered endpoint identity and QoS, preserving unknown values."""
    return [
        {
            "node": x.node_name,
            "namespace": x.node_namespace,
            "type": x.topic_type,
            "reliability": x.qos_profile.reliability.name,
            "durability": x.qos_profile.durability.name,
            "depth": x.qos_profile.depth,
        }
        for x in bounded(items, 16)
    ]


def health_sample(msg: Any, profile: str) -> dict[str, Any]:
    """Retain current typed status independently of measured motion streams."""
    stamp = msg.header.stamp
    if profile == "robotiq":
        data = {
            key: getattr(msg, key)
            for key in (
                "object_detection",
                "motor_current",
                "gripper_fault",
                "gripper_fault_severity",
            )
        }
        if not math.isfinite(data["motor_current"]) or any(
            type(data[key]) is not int or not 0 <= data[key] <= 255
            for key in ("object_detection", "gripper_fault", "gripper_fault_severity")
        ):
            raise ValueError("Expected finite current and uint8 gripper status")
    else:
        names = bounded(msg.current_errors.get_fields_and_field_types(), 128)
        data = {
            "robot_mode": int(msg.robot_mode),
            "current_errors": {key: getattr(msg.current_errors, key) for key in names},
        }
        if not names or any(
            type(v) is not bool for v in data["current_errors"].values()
        ):
            raise ValueError("Expected named boolean current_errors")
    return {
        "data": data,
        "header": {
            "sec": stamp.sec,
            "nanosec": stamp.nanosec,
            "frame_id": msg.header.frame_id,
        },
        "received_monotonic_ns": time.monotonic_ns(),
    }


def collect_status(
    node: Any, executor: Any, profile: str, budget: float
) -> dict[str, Any]:
    """Collect whitelisted listing/parameter services, graph and latest health.

    Every service is read-only. No action client or command writer is created.
    Samples continue through the same executor during the shared time budget.
    """
    from rclpy.action import (
        get_action_client_names_and_types_by_node,
        get_action_server_names_and_types_by_node,
    )
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message, get_service

    if (
        type(budget) not in (int, float)
        or not math.isfinite(budget)
        or not 0 < budget <= 5
    ):
        raise ValueError("Expected finite diagnostic budget in (0, 5]")
    started = time.monotonic_ns()
    stop = time.monotonic() + budget
    result: dict[str, Any] = {
        "errors": {},
        "services": {},
        "health": {
            "topic": HEALTH[profile][0],
            "type": HEALTH[profile][1],
            "sample": None,
            "error": "No matching health publisher discovered",
        },
    }
    clients, pending = [], {}
    subscription = None
    try:
        queries = {
            "/controller_manager/" + name: ("controller_manager_msgs/srv/" + kind, ())
            for name, kind in (
                ("list_controllers", "ListControllers"),
                ("list_hardware_components", "ListHardwareComponents"),
                ("list_hardware_interfaces", "ListHardwareInterfaces"),
            )
        }
        queries.update(
            {
                target + "/get_parameters": ("rcl_interfaces/srv/GetParameters", names)
                for target, names in PARAMETERS[profile].items()
            }
        )
        for name, (kind, names) in queries.items():
            try:
                service = get_service(kind)
                client = node.create_client(service, name)
                clients.append(client)
                request = service.Request()
                if names:
                    request.names = list(names)
                pending[name] = [client, request, None, names]
            except Exception as exc:
                result["errors"][name] = str(exc)[:512]

        def callback(msg: Any) -> None:
            try:
                result["health"]["sample"] = health_sample(msg, profile)
                result["health"]["error"] = None
            except (ValueError, TypeError, AttributeError) as exc:
                result["health"]["sample"] = None
                result["health"]["error"] = str(exc)[:512]

        next_health_discovery = 0.0
        while time.monotonic() < stop:
            if subscription is None and time.monotonic() >= next_health_discovery:
                next_health_discovery = time.monotonic() + 0.25
                topic, kind = HEALTH[profile]
                try:
                    infos = node.get_publishers_info_by_topic(topic)
                    if any(x.topic_type != kind for x in infos):
                        raise ValueError("Conflicting health publisher type")
                    if infos:
                        subscription = node.create_subscription(
                            get_message(kind),
                            topic,
                            callback,
                            QoSProfile(
                                depth=5,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE,
                            ),
                        )
                except Exception as exc:
                    result["health"]["error"] = str(exc)[:512]
            for name, (client, request, future, names) in list(pending.items()):
                try:
                    if future is None and client.service_is_ready():
                        future = client.call_async(request)
                        pending[name][2] = future
                    if future is None or not future.done():
                        continue
                    response = future.result()
                    if name.endswith("list_controllers"):
                        data = [
                            {
                                "name": x.name,
                                "type": x.type,
                                "state": x.state,
                                "claimed": bounded(x.claimed_interfaces, 128),
                            }
                            for x in bounded(response.controller, 32)
                        ]
                    elif name.endswith("list_hardware_components"):
                        data = [
                            {
                                "name": x.name,
                                "state": x.state.label,
                                "state_id": x.state.id,
                                "command": interfaces(x.command_interfaces),
                            }
                            for x in bounded(response.component, 8)
                        ]
                    elif name.endswith("list_hardware_interfaces"):
                        data = interfaces(response.command_interfaces)
                    else:
                        if len(response.values) != len(names):
                            raise ValueError(
                                "Expected one value per requested parameter"
                            )
                        data = dict(zip(names, (parameter(x) for x in response.values)))
                    result["services"][name] = data
                    del pending[name]
                except Exception as exc:
                    result["errors"][name] = str(exc)[:512]
                    del pending[name]
            executor.spin_once(timeout_sec=min(0.01, max(0.0, stop - time.monotonic())))
        for name in pending:
            result["errors"][name] = (
                "Discovery/response timeout within shared diagnostic budget"
            )
        topics = (*STATE_TOPICS[profile], HEALTH[profile][0])
        if profile == "fr3":
            topics += (COMMAND_TOPIC, BRIDGE_TOPIC)
        result["topics"] = {
            topic: {
                "publishers": endpoints(node.get_publishers_info_by_topic(topic)),
                "subscriptions": endpoints(node.get_subscriptions_info_by_topic(topic)),
            }
            for topic in topics
        }
        nodes = bounded(node.get_node_names_and_namespaces(), 128)
        result["nodes"] = nodes
        result["action_servers"], result["action_clients"] = [], []
        for name, namespace in nodes:
            for key, getter in (
                ("action_servers", get_action_server_names_and_types_by_node),
                ("action_clients", get_action_client_names_and_types_by_node),
            ):
                for endpoint, kinds in bounded(getter(node, name, namespace), 16):
                    if endpoint in GRIPPER_ACTIONS:
                        result[key].append(
                            {
                                "node": name,
                                "namespace": namespace,
                                "endpoint": endpoint,
                                "types": bounded(kinds, 4),
                            }
                        )
                        bounded(result[key], 32)
    except Exception as exc:
        result["errors"]["collection"] = str(exc)[:512]
    finally:
        if subscription is not None:
            node.destroy_subscription(subscription)
        for client in clients:
            node.destroy_client(client)
    result["elapsed_s"] = (time.monotonic_ns() - started) / 1e9
    result["collected_monotonic_ns"] = time.monotonic_ns()
    return result
