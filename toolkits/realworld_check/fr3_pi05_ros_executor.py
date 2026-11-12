# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""ROS Jazzy 3.12 process: supervised π0.5 request execution on the FR3.

STDIN: JSON lines begin_episode/chunk/end_episode/quit.
STDOUT: JSON events, including actual ROS publish counts.
Requires ROS_DOMAIN_ID=70, exclusive motion publisher, fresh measured pose.
A neutral request/ROS watchdog is NOT a safety-rated stop. Operator must keep
the approved Franka enable/E-stop available throughout physical execution.
"""

from __future__ import annotations

import json
import math
import os
import select
import sys
import time

# Works both as a directly launched ROS Python script and a package import.
if __package__:
    from .fr3_pi05_request_adapter import (
        GripperRequestFilter, identity_arm_request, validate_action_chunk,
    )
else:
    from fr3_pi05_request_adapter import (
        GripperRequestFilter, identity_arm_request, validate_action_chunk,
    )

ARM_TOPIC = "/franka_controller/target_cartesian_velocity_percent"
GRIPPER_TOPIC = "/gripper_client/target_gripper_width_percent"
POSE_TOPIC = "/franka_robot_state_broadcaster/current_pose"
NAME = "rlinf_fr3_pi05_executor"
PERIOD_S = 0.1  # SFT policy labels at 10 Hz
LEGACY_GAIN = 0.15  # existing experimental motion limit; NOT trained mapping
POSE_TIMEOUT_S = 0.25
MAX_EPISODE_S = 20.0
MAX_TRAVEL_M = 0.08


def validate_chunk(chunk, *, observation_ns: int, issued_ns: int, now_ns: int) -> list[list[float]]:
    if not isinstance(chunk, list) or len(chunk) != 10:
        raise ValueError("Expected exactly ten 7D requests")
    if (not all(type(t) is int for t in (observation_ns, issued_ns))
            or not 0 < observation_ns <= issued_ns <= now_ns
            or now_ns - observation_ns > 1_000_000_000
            or now_ns - issued_ns > 300_000_000):
        raise ValueError("Stale observation or issued request")
    return validate_action_chunk(chunk)


def main() -> int:
    if os.environ.get("ROS_DOMAIN_ID") != "70":
        raise RuntimeError("This executor requires the explicit FR3 domain 70")
    from geometry_msgs.msg import PoseStamped, Twist
    from std_msgs.msg import Float32
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = Node(NAME, enable_rosout=False, start_parameter_services=False)
    pose = None
    pose_at = 0.
    origin = None
    started = 0.
    batch = []
    next_action = 0
    next_at = 0.
    last_gripper = None
    publisher = None
    gripper_pub = None
    total_commands = 0
    # Identity mapping must be explicitly enabled after physical interface review.
    mapping_mode = os.environ.get("FR3_PI05_MAPPING_MODE", "legacy_limited")
    if mapping_mode not in ("identity_verified", "legacy_limited"):
        raise RuntimeError("Unknown FR3 request mapping mode")
    gain = 1.0 if mapping_mode == "identity_verified" else LEGACY_GAIN
    gripper_filter = GripperRequestFilter(confirmations=3, min_goal_interval_s=0.8)

    def emit(**fields):
        print(json.dumps(fields, allow_nan=False), flush=True)

    def pose_cb(msg):
        nonlocal pose, pose_at
        p = msg.pose.position
        pose = (float(p.x), float(p.y), float(p.z))
        pose_at = time.monotonic()

    node.create_subscription(PoseStamped, POSE_TOPIC, pose_cb, qos_profile_sensor_data)

    def fresh_pose():
        if (pose is None or not all(map(math.isfinite, pose))
                or time.monotonic() - pose_at > POSE_TIMEOUT_S):
            raise RuntimeError("FR3 pose missing or stale")
        if origin is not None and math.dist(origin, pose) > MAX_TRAVEL_M:
            raise RuntimeError("Measured FR3 displacement exceeded 8 cm limit")
        if origin is not None and time.monotonic() - started > MAX_EPISODE_S:
            raise RuntimeError("Supervised episode time limit reached")

    def exclusive():
        publishers = node.get_publishers_info_by_topic(ARM_TOPIC)
        if len(publishers) != 1 or publishers[0].node_name != NAME:
            raise RuntimeError("SpaceMouse/other publisher detected: exclusive control lost")
        if not node.get_subscriptions_info_by_topic(ARM_TOPIC):
            raise RuntimeError("No FR3 controller subscriber")
        if gripper_pub is not None:
            writers = node.get_publishers_info_by_topic(GRIPPER_TOPIC)
            if len(writers) != 1 or writers[0].node_name != NAME:
                raise RuntimeError("Competing gripper publisher detected")
            if not node.get_subscriptions_info_by_topic(GRIPPER_TOPIC):
                raise RuntimeError("Robotiq bridge subscriber missing")

    def neutral():
        if publisher is not None:
            publisher.publish(Twist())

    try:
        # DDS discovery is asynchronous (especially with the NUC's ROS
        # Humble -> workstation Jazzy endpoints). Wait for both known
        # subscribers instead of failing the instant the first pose arrives.
        deadline = time.monotonic() + 10.0
        arm_subs = []
        gripper_subs = []
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            # Never accept an active SpaceMouse or other competing publisher.
            if node.get_publishers_info_by_topic(ARM_TOPIC):
                raise RuntimeError("Stop SpaceMouse arm publisher before policy rollout")
            if node.get_publishers_info_by_topic(GRIPPER_TOPIC):
                raise RuntimeError("Stop SpaceMouse gripper publisher before policy rollout")
            arm_subs = node.get_subscriptions_info_by_topic(ARM_TOPIC)
            gripper_subs = node.get_subscriptions_info_by_topic(GRIPPER_TOPIC)
            if pose is not None and arm_subs and gripper_subs:
                break
        fresh_pose()
        if not arm_subs or not gripper_subs:
            raise RuntimeError(
                "ROS discovery timeout: arm subscribers="
                f"{len(arm_subs)}, gripper subscribers={len(gripper_subs)}. "
                "No robot commands sent."
            )
        publisher = node.create_publisher(Twist, ARM_TOPIC, 10)
        gripper_pub = node.create_publisher(Float32, GRIPPER_TOPIC, 10)
        # ROS graph propagation.
        for _ in range(12):
            rclpy.spin_once(node, timeout_sec=0.05)
        exclusive()
        emit(event="ready", arm_topic=ARM_TOPIC, gripper_topic=GRIPPER_TOPIC,
             mapping_mode=mapping_mode, arm_gain=gain)

        while True:
            rclpy.spin_once(node, timeout_sec=0.01)
            fresh_pose()
            exclusive()
            now = time.monotonic()
            if batch and now >= next_at:
                if next_action >= len(batch):
                    neutral()
                    batch = []
                    emit(event="chunk_done", commands_sent=total_commands)
                else:
                    values = batch[next_action]
                    # Raw normalized SpaceMouse request axis order (not SI velocity).
                    arm = identity_arm_request(values)
                    sent_arm = [x * gain for x in arm]
                    msg = Twist()
                    msg.linear.x, msg.linear.y, msg.linear.z = sent_arm[:3]
                    msg.angular.x, msg.angular.y, msg.angular.z = sent_arm[3:]
                    publisher.publish(msg)
                    total_commands += 1
                    g = values[6]
                    gripper_event = gripper_filter.step(g, now=time.monotonic())
                    if gripper_event != "hold":
                        gripper_msg = Float32()
                        gripper_msg.data = 0. if gripper_event == "close" else 1.
                        gripper_pub.publish(gripper_msg)
                    emit(event="action", command_index=next_action,
                         requested_7d=values, sent_arm_6d=sent_arm,
                         gripper_event=gripper_event,
                         tcp_xyz=pose, monotonic_ns=time.monotonic_ns(),
                         commands_sent=total_commands)
                    next_action += 1
                    next_at += PERIOD_S
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:  # parent died: never continue policy control
                    break
                item = json.loads(line)
                op = item.get("op")
                if op == "begin_episode":
                    if origin is not None or batch:
                        raise RuntimeError("Episode already active")
                    fresh_pose()
                    origin = tuple(pose)
                    started = time.monotonic()
                    last_gripper = None
                    gripper_filter.reset()
                    total_commands = 0
                    emit(event="episode_started", tcp_xyz=origin)
                elif op == "chunk":
                    if origin is None or batch:
                        raise RuntimeError("No active episode or previous chunk not finished")
                    batch = validate_chunk(
                        item.get("values"),
                        observation_ns=item.get("observation_ns"),
                        issued_ns=item.get("issued_ns"),
                        now_ns=time.monotonic_ns(),
                    )
                    next_action = 0
                    next_at = time.monotonic()
                elif op == "end_episode":
                    neutral()
                    batch = []
                    origin = None
                    emit(event="episode_ended", commands_sent=total_commands)
                elif op == "quit":
                    break
                else:
                    raise RuntimeError("Unexpected executor operation")
        return 0
    except Exception as exc:
        emit(event="fault", reason=str(exc), commands_sent=total_commands)
        return 2
    finally:
        if publisher is not None:
            for _ in range(5):
                neutral()
                rclpy.spin_once(node, timeout_sec=0.02)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
