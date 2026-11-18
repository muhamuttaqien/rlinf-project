# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deterministic ROS/hardware boundary for the real calibration state machine."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

from toolkits.realworld_check.fr3_mapping_calibration import (
    ARM_TOPIC,
    DIAG_FIELDS,
    GRIPPER_TOPIC,
    NAME,
    Blocked,
    Console,
    Telemetry,
    check_endpoints,
    multiply,
    sha256,
)
from toolkits.realworld_check.fr3_mapping_calibration_ros import RosRuntime


def pose_message(value=None):
    value = value or [0, 0, 0, 0, 0, 0, 1]
    return {
        "header": {"frame_id": "test_base", "stamp": {"sec": 0, "nanosec": 0}},
        "pose": {
            "position": dict(zip(("x", "y", "z"), value[:3])),
            "orientation": dict(zip(("x", "y", "z", "w"), value[3:])),
        },
    }


def deployment_file(directory: Path) -> Path:
    """Write clearly synthetic installation evidence for offline tests only."""
    directory.mkdir(parents=True, exist_ok=True)
    binary, config, params = [
        directory / name
        for name in ("FAKE_CONTROLLER.so", "config.json", "params.json")
    ]
    binary.write_bytes(b"OFFLINE FAKE HARDWARE; NOT A CONTROLLER BINARY\n")
    params.write_text('{"fixture_only":true}\n')
    config.write_text(
        json.dumps(
            {
                "schema": "fr3_mapping_config_manifest_v1",
                "files": [{"path": params.name, "sha256": sha256(params)}],
            }
        )
    )
    record = {
        "schema": "fr3_mapping_calibration_deployment_v1",
        "operator": "OFFLINE TEST FIXTURE",
        "base_frame": "test_base",
        "pose_frame": "test_base",
        "tool_frame": "test_peg",
        "diagnostic_logger": "joint_impedance_ik_controller",
        "gripper_joint": "test_robotiq_joint",
        "gripper_bridge_node": "test_gripper_bridge",
        "gripper_status_topic": "/fake_gripper/_action/status",
        "translation_limit_m": 0.01,
        "rotation_limit_rad": 0.01,
        "base_to_pose": [0, 0, 0, 0, 0, 0, 1],
        "ee_to_flange": [0, 0, 0, 0, 0, 0, 1],
        "ee_to_tcp": [0, 0, 0.2, 0, 0, 0, 1],
        "controller_binary": {"path": binary.name, "sha256": sha256(binary)},
        "controller_config": {"path": config.name, "sha256": sha256(config)},
    }
    record.update(
        dict.fromkeys(
            (
                "live_installation_matched",
                "frames_calibrated",
                "hardware_stop_checked",
                "workspace_clear",
                "historical_errors_reviewed",
            ),
            True,
        )
    )
    path = directory / "deployment.json"
    path.write_text(json.dumps(record))
    return path


class FakeConsole(Console):
    """Emulate typed consent/stop at the terminal boundary, without ROS."""

    def __init__(
        self, runtime, *, stop_after_ns=None, approve=True, confirmation_delay_ns=0
    ):
        super().__init__()
        self.runtime = runtime
        self.stop_after_ns = stop_after_ns
        self.approve = approve
        self.confirmation_delay_ns = confirmation_delay_ns
        self.tokens = []

    def confirm(self, token):
        self.tokens.append(token)
        self.runtime.clock += self.confirmation_delay_ns
        if not self.approve:
            raise Blocked("Operator canceled")

    def stopped(self):
        return self.interrupted or (
            self.stop_after_ns is not None
            and self.runtime.enabled_at is not None
            and self.runtime.now_ns() - self.runtime.enabled_at >= self.stop_after_ns
        )

    def active(self, plan):
        pass


class FakePublisher:
    """Record actual message fields at the vendor/ROS publication edge."""

    def __init__(self, runtime, topic):
        self.runtime = runtime
        self.topic_name = topic

    def publish(self, message):
        r = self.runtime
        if self.topic_name == ARM_TOPIC:
            r.request = [
                getattr(part, axis)
                for part in (message.linear, message.angular)
                for axis in ("x", "y", "z")
            ]
            r.arm_messages.append((r.now_ns(), list(r.request)))
            r.diag["rx"] += 1
            r.diag["moving_rx"] += int(any(r.request))
        else:
            r.gripper_messages.append((r.now_ns(), message.data))
            r.gripper_position = 0.8 if message.data == 1 else 0.0


class FakeRuntime:
    """Drive the production publisher methods against deterministic fake sensors."""

    backend = "fake"
    clock_id = "OFFLINE_FAKE_CLOCK"
    publish_arm = RosRuntime.publish_arm
    publish_gripper = RosRuntime.publish_gripper

    def __init__(self, deployment, *, gripper=False):
        self.deployment = deployment
        self.gripper = gripper
        self.telemetry = Telemetry()
        self.journal = None
        self.clock = 1_000_000_000
        self.enabled_at = None
        self.origin = None
        self.arm_pub = None
        self.gripper_pub = None
        self.arm_messages = []
        self.gripper_messages = []
        self.request = [0.0] * 6
        self.measured_pose = [0.0] * 6 + [1.0]
        self.gripper_position = 0.0
        self.diag = dict.fromkeys(DIAG_FIELDS, 0)
        self.last_diag = 0
        self.spins = 0
        self.omit = set()
        self.mutate = lambda _: None
        self.competing = False
        self.extra_subscriber = False
        self.missing_subscriber = False
        self.construct_delay_ns = 0
        self.hook = lambda _: None
        self.float_type = lambda: SimpleNamespace(data=0.0)

    def twist_type(self):
        self.clock += self.construct_delay_ns
        return SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        )

    def now_ns(self):
        return self.clock

    def spin(self):
        self.clock += 2_000_000
        self.spins += 1
        self.hook(self)
        for i in range(3):
            self.measured_pose[i] += self.request[i] * 0.002 * 0.01
        rot = self.request[3:]
        length = math.sqrt(sum(v * v for v in rot))
        if length:
            theta = length * 0.002 * 0.02
            q = tuple(v / length * math.sin(theta / 2) for v in rot) + (
                math.cos(theta / 2),
            )
            self.measured_pose[3:] = multiply(tuple(self.measured_pose[3:]), q)
        if any(self.request):
            for key in (
                "ik_attempt",
                "ik_valid",
                "gate_ticks",
                "ref_advanced_ticks",
                "effort_write_ticks",
            ):
                self.diag[key] += 1
            for key in DIAG_FIELDS[9:]:
                self.diag[key] = 0.001
        joints = {
            "name": [f"fr3_joint{i}" for i in range(1, 8)],
            "position": [self.measured_pose[0]] * 7,
            "velocity": [0.0] * 7,
            "effort": [0.01] * 7,
        }
        messages = {
            "pose": pose_message(self.measured_pose),
            "measured_joints": joints,
            "desired_joints": joints,
            "controller": {
                "state": "active",
                "claimed_interfaces": [f"fr3_joint{i}/effort" for i in range(1, 8)],
            },
            "robot_state": {
                "robot_mode": 2,
                "current_errors": {"joint_reflex": False},
                "last_motion_errors": {"joint_reflex": False},
                "control_command_success_rate": 1.0,
                "f_t_ee": pose_message(),
            },
            "gripper_state": {
                "name": [self.deployment["record"]["gripper_joint"]],
                "position": [self.gripper_position],
            },
            "gripper_status": {"status_list": [{"status": 4}]},
        }
        if self.clock - self.last_diag >= 1_000_000_000:
            messages["diagnostics"] = dict(self.diag)
            self.last_diag = self.clock
        self.mutate(messages)
        for name, message in messages.items():
            if name not in self.omit:
                self.telemetry.receive(name, message, self.now_ns())
                if self.journal and (self.spins % 5 == 0 or name == "diagnostics"):
                    self.journal.add(
                        "telemetry",
                        {
                            "source": name,
                            "received_monotonic_ns": self.now_ns(),
                            "message": message,
                        },
                    )
        if self.spins % 25 == 0:
            time.sleep(0)

    def enable_publishers(self):
        self.enabled_at = self.now_ns()
        self.arm_pub = FakePublisher(self, ARM_TOPIC)
        if self.gripper:
            self.gripper_pub = FakePublisher(self, GRIPPER_TOPIC)

    def check_graph(self, *, own):
        def endpoint(name, cls):
            return SimpleNamespace(node_name=name, topic_type=cls)

        for topic, cls, peer in (
            (ARM_TOPIC, "geometry_msgs/msg/Twist", "joint_impedance_ik_controller"),
            (GRIPPER_TOPIC, "std_msgs/msg/Float32", "test_gripper_bridge"),
        ):
            writers = (
                [endpoint(NAME, cls)]
                if own and (topic == ARM_TOPIC or self.gripper)
                else []
            )
            if self.competing:
                writers.append(endpoint("other_operator", cls))
            readers = [] if self.missing_subscriber else [endpoint(peer, cls)]
            if self.extra_subscriber:
                readers.append(endpoint("recorder", cls))
            check_endpoints(
                writers,
                readers,
                own=own and (topic == ARM_TOPIC or self.gripper),
                topic_type=cls,
                peer=peer,
            )
