# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""ROS endpoints for the single-case FR3 calibration state machine."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

from toolkits.realworld_check.fr3_mapping_calibration import (
    ARM_TOPIC,
    DIAG_FIELDS,
    GRIPPER_TOPIC,
    NAME,
    Blocked,
    Telemetry,
    check_endpoints,
    publication_guard,
)


def json_message(value, path: str = "") -> tuple[object, list[str]]:
    """Retain optional nonfinite ROS fields as null with explicit field paths."""
    if isinstance(value, dict):
        result, missing = {}, []
        for key, child in value.items():
            result[key], paths = json_message(child, f"{path}.{key}")
            missing.extend(paths)
        return result, missing
    if isinstance(value, (list, tuple)):
        result, missing = [], []
        for index, child in enumerate(value):
            item, paths = json_message(child, f"{path}[{index}]")
            result.append(item)
            missing.extend(paths)
        return result, missing
    if isinstance(value, float) and not math.isfinite(value):
        return None, [path]
    return value, []


def select_gripper_joint(message: dict, joint: str) -> dict | None:
    """Select the physical joint; unrelated messages never refresh its watchdog."""
    names = message.get("name", [])
    if joint not in names:
        return None
    if names.count(joint) != 1:
        raise Blocked(f"Expected one physical gripper joint: {joint}")
    index = names.index(joint)
    positions = message.get("position", [])
    if (
        len(positions) <= index
        or type(positions[index]) not in (float, int)
        or not math.isfinite(positions[index])
    ):
        raise Blocked(f"Missing or nonfinite measured gripper position: {joint}")
    selected = {
        "header": message.get("header", {}),
        "name": [joint],
        "position": [positions[index]],
    }
    for field in ("velocity", "effort"):
        values = message.get(field, [])
        selected[field] = [values[index]] if len(values) > index else []
    return selected


class RosRuntime:
    """Own read-only ROS contexts and, after consent, command publishers.

    Robot state is sampled at at most 100 Hz for evidence. Message headers and
    host receipts are retained separately; neither is a command acknowledgement.
    """

    backend = "ros"

    def __init__(self, deployment: dict, *, gripper: bool) -> None:
        self.deployment = deployment
        self.gripper = gripper
        self.telemetry = Telemetry()
        self.journal = None
        self.contexts = []
        self.executors = []
        self.nodes = []
        self.arm_pub = None
        self.gripper_pub = None
        self.error = None
        self.future = None
        self.next_control = 0
        self.last_robot_record = 0
        self.received_robot_messages = 0
        self.ignored_gripper_messages = 0
        self.origin = None
        self.clock_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def __enter__(self) -> RosRuntime:
        import rclpy
        from controller_manager_msgs.srv import ListControllers
        from geometry_msgs.msg import PoseStamped, Twist
        from rcl_interfaces.msg import Log
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_action_status_default, qos_profile_sensor_data
        from rosidl_runtime_py.convert import message_to_ordereddict
        from rosidl_runtime_py.utilities import get_message
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float32

        self.convert = message_to_ordereddict
        self.twist_type = Twist
        self.float_type = Float32
        self.control_type = ListControllers
        self.qos = qos_profile_sensor_data
        raw = self.deployment["record"]
        if self.gripper and not isinstance(raw.get("gripper_status_topic"), str):
            raise Blocked(
                "Gripper cases require the inspected domain-71 action status topic"
            )
        try:
            for domain in (70, 71):
                context = Context()
                rclpy.init(context=context, domain_id=domain)
                self.contexts.append(context)
                node = Node(
                    NAME,
                    context=context,
                    enable_rosout=False,
                    start_parameter_services=False,
                )
                self.nodes.append(node)
                executor = SingleThreadedExecutor(context=context)
                self.executors.append(executor)
                executor.add_node(node)
            arm, grip = self.nodes
            broadcaster = "/franka_robot_state_broadcaster"
            for source, suffix, cls in (
                ("pose", "current_pose", PoseStamped),
                ("measured_joints", "measured_joint_states", JointState),
                ("desired_joints", "desired_joint_states", JointState),
                (
                    "robot_state",
                    "robot_state",
                    get_message("franka_msgs/msg/FrankaRobotState"),
                ),
            ):
                topic = f"{broadcaster}/{suffix}"
                arm.create_subscription(
                    cls, topic, self._receiver(source, topic), self.qos
                )
            arm.create_subscription(Log, "/rosout", self._diagnostics, self.qos)
            grip.create_subscription(
                JointState,
                "/joint_states",
                self._receiver("gripper_state", "/joint_states"),
                self.qos,
            )
            if self.gripper:
                topic = raw["gripper_status_topic"]
                grip.create_subscription(
                    get_message("action_msgs/msg/GoalStatusArray"),
                    topic,
                    self._receiver("gripper_status", topic),
                    qos_profile_action_status_default,
                )
            self.client = arm.create_client(
                ListControllers, "/controller_manager/list_controllers"
            )
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _record(self, source: str, topic: str, message: dict, **metadata) -> None:
        now = self.now_ns()
        self.telemetry.receive(source, message, now)
        if self.journal:
            self.journal.add(
                "telemetry",
                {
                    "source": source,
                    "topic": topic,
                    "domain": 71 if source.startswith("gripper_") else 70,
                    "received_monotonic_ns": now,
                    "received_unix_ns": time.time_ns(),
                    "message": message,
                    **metadata,
                },
            )

    def _receiver(self, source: str, topic: str):
        def receive(message):
            try:
                if source == "robot_state":
                    self.received_robot_messages += 1
                    if self.now_ns() - self.last_robot_record < 10_000_000:
                        return
                    self.last_robot_record = self.now_ns()
                payload, nonfinite = json_message(self.convert(message))
                if source == "gripper_state":
                    payload = select_gripper_joint(
                        payload, self.deployment["record"]["gripper_joint"]
                    )
                    if payload is None:
                        self.ignored_gripper_messages += 1
                        return
                self._record(
                    source,
                    topic,
                    payload,
                    nonfinite_fields=nonfinite,
                    robot_state_receipts=self.received_robot_messages,
                )
            except Exception as exc:
                self.error = exc

        return receive

    def _diagnostics(self, message) -> None:
        if (
            message.name != self.deployment["record"]["diagnostic_logger"]
            or "FR3_DIAG_V1 " not in message.msg
        ):
            return
        try:
            values = {
                key: float(value)
                for key, value in re.findall(
                    r"([a-z][a-z0-9_]*)=([-+0-9.eE]+)", message.msg
                )
            }
            if any(key not in values for key in DIAG_FIELDS):
                raise Blocked("Controller diagnostics lack required counters/peaks")
            self._record("diagnostics", "/rosout", values, raw=self.convert(message))
        except Exception as exc:
            self.error = exc

    def now_ns(self) -> int:
        """Read the host monotonic clock used by all local evidence streams."""
        return time.monotonic_ns()

    def spin(self) -> None:
        """Process bounded callback work and read-only controller status requests."""
        for _ in range(4):
            for executor in self.executors:
                executor.spin_once(timeout_sec=0)
        now = self.now_ns()
        if self.future is not None and self.future.done():
            try:
                response = self.future.result()
                controllers = [
                    c
                    for c in response.controller
                    if c.name == "joint_impedance_ik_controller"
                ]
                if len(controllers) != 1:
                    raise Blocked(
                        "Expected one joint impedance controller in read-only service response"
                    )
                self._record(
                    "controller",
                    "/controller_manager/list_controllers",
                    self.convert(controllers[0]),
                )
            except Exception as exc:
                self.error = exc
            self.future = None
        if (
            now >= self.next_control
            and self.future is None
            and self.client.service_is_ready()
        ):
            self.future = self.client.call_async(self.control_type.Request())
            self.next_control = now + 200_000_000
        if self.error is not None:
            raise Blocked(f"ROS telemetry capture failed: {self.error}")
        time.sleep(0.001)

    def check_graph(self, *, own: bool) -> None:
        """Fail closed on competing writers, wrong types or extra subscribers."""
        arm = self.nodes[0]
        check_endpoints(
            arm.get_publishers_info_by_topic(ARM_TOPIC),
            arm.get_subscriptions_info_by_topic(ARM_TOPIC),
            own=own,
            topic_type="geometry_msgs/msg/Twist",
            peer="joint_impedance_ik_controller",
        )
        check_endpoints(
            arm.get_publishers_info_by_topic(GRIPPER_TOPIC),
            arm.get_subscriptions_info_by_topic(GRIPPER_TOPIC),
            own=own and self.gripper,
            topic_type="std_msgs/msg/Float32",
            peer=self.deployment["record"].get("gripper_bridge_node"),
        )
        if self.gripper:
            publishers = self.nodes[1].get_publishers_info_by_topic(
                self.deployment["record"]["gripper_status_topic"]
            )
            if not publishers:
                from toolkits.realworld_check.fr3_mapping_calibration import NotReady

                raise NotReady("Robotiq action status publisher unavailable")
            if (
                len(publishers) != 1
                or publishers[0].topic_type != "action_msgs/msg/GoalStatusArray"
            ):
                raise Blocked(
                    "Expected exactly one typed Robotiq action status publisher"
                )

    def enable_publishers(self) -> None:
        """Create publishers only after the caller's exact interactive consent."""
        if self.deployment.get("read_only"):
            raise Blocked("Read-only preflight cannot create command publishers")
        arm = self.nodes[0]
        self.arm_pub = arm.create_publisher(self.twist_type, ARM_TOPIC, 1)
        if self.arm_pub.topic_name != ARM_TOPIC:
            raise Blocked("Calibration arm topic was remapped")
        if self.gripper:
            self.gripper_pub = arm.create_publisher(self.float_type, GRIPPER_TOPIC, 10)
            if self.gripper_pub.topic_name != GRIPPER_TOPIC:
                raise Blocked("Calibration gripper topic was remapped")

    def publish_arm(
        self, values: list, *, deadline_ns: int | None, last_ns: int | None, stop
    ) -> int:
        """Recheck sample expiry and stop after constructing the ROS message."""
        message = self.twist_type()
        message.linear.x, message.linear.y, message.linear.z = values[:3]
        message.angular.x, message.angular.y, message.angular.z = values[3:]
        if any(values):
            self.check_graph(own=True)
            self.telemetry.check(
                self.deployment, self.now_ns(), gripper=self.gripper, origin=self.origin
            )
            publication_guard(self.now_ns(), deadline_ns, last_ns, stop)
        self.arm_pub.publish(message)
        return self.now_ns()

    def publish_gripper(
        self, value: float, *, deadline_ns: int, last_ns: int, stop
    ) -> int:
        """Emit at most the authorized filter goal, never cancel an in-flight goal."""
        message = self.float_type()
        message.data = value
        self.check_graph(own=True)
        self.telemetry.check(
            self.deployment, self.now_ns(), gripper=True, origin=self.origin
        )
        publication_guard(self.now_ns(), deadline_ns, last_ns, stop)
        self.gripper_pub.publish(message)
        return self.now_ns()

    def __exit__(self, *_args) -> None:
        errors = []
        for executor in self.executors:
            try:
                executor.shutdown(timeout_sec=0.5)
            except Exception as exc:
                errors.append(str(exc))
        for node in self.nodes:
            try:
                node.destroy_node()
            except Exception as exc:
                errors.append(str(exc))
        for context in self.contexts:
            try:
                context.try_shutdown()
            except Exception as exc:
                errors.append(str(exc))
        self.executors.clear()
        self.nodes.clear()
        self.contexts.clear()
        if errors:
            raise Blocked(f"ROS calibration cleanup failed: {errors}")
