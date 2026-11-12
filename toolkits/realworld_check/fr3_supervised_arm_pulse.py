# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""First supervised FR3 arm-only control qualification (NOT autonomous rollout).

Default: READ-ONLY ROS graph/pose check, creates NO publishers.
--execute: single bounded X-axis test pulse with operator confirmation,
exclusive ROS topic ownership, live-pose watchdog, and immediate neutral
request at exit. The existing Franka hardware E-stop remains essential:
controller timeout / neutral request is NOT a safety-rated stop.

No gripper API, no SpaceMouse HID access, no policy inference, no looping
rollouts, no recovery/reset calls. Do not use near obstacles or people.
"""

from __future__ import annotations

import argparse
import math
import os
import time

MOTION_TOPIC = "/franka_controller/target_cartesian_velocity_percent"
POSE_TOPIC = "/franka_robot_state_broadcaster/current_pose"
CONTROLLER = "joint_impedance_ik_controller"
NODE_NAME = "rlinf_fr3_supervised_arm_pulse"

AXIS = "linear_x"
MAX_REQUEST = 0.05  # normalized operator request, not metres per second
DEFAULT_REQUEST = 0.03
MAX_PULSE_S = 0.15
POSE_MAX_AGE_S = 0.2
MAX_OBSERVED_DISPLACEMENT_M = 0.01


class Blocked(RuntimeError):
    """Supervised motion is not permitted."""


def qualify_graph(publishers: list, subscribers: list, *,
                  own_publisher: bool = False) -> None:
    """Fail closed on any competing command writer or missing controller."""
    if own_publisher:
        if len(publishers) != 1 or publishers[0].node_name != NODE_NAME:
            raise Blocked("Competing publisher appeared: cancel physical pulse")
    elif publishers:
        names = [p.node_name for p in publishers]
        raise Blocked(f"Existing command publisher(s): {names}; stop SpaceMouse first")
    if not any(s.node_name == CONTROLLER for s in subscribers):
        raise Blocked("Active FR3 joint_impedance_ik_controller subscriber not found")


def qualify_pose(pose: tuple[float, float, float] | None, *,
                 receipt_monotonic: float | None, now: float,
                 origin: tuple[float, float, float] | None = None) -> None:
    if pose is None or receipt_monotonic is None:
        raise Blocked("No measured FR3 pose")
    if len(pose) != 3 or any(not math.isfinite(v) for v in pose):
        raise Blocked("Nonfinite FR3 pose")
    if now < receipt_monotonic or now - receipt_monotonic > POSE_MAX_AGE_S:
        raise Blocked("Stale FR3 pose")
    if origin is not None:
        travel = math.dist(pose, origin)
        if travel > MAX_OBSERVED_DISPLACEMENT_M:
            raise Blocked(f"Displacement {travel:.4f}m exceeded supervised qualification limit")


def validate_pulse_request(value: float, duration: float) -> None:
    if (not math.isfinite(value) or not 0 < abs(value) <= MAX_REQUEST
            or not math.isfinite(duration) or not 0 < duration <= MAX_PULSE_S):
        raise Blocked("Only a single <=0.05 normalized X request, <=0.15s pulse is allowed")


def run(*, execute: bool, request: float, duration: float) -> int:
    validate_pulse_request(request, duration)
    if os.environ.get("ROS_DOMAIN_ID") != "70":
        raise Blocked("Set ROS_DOMAIN_ID=70 (actual FR3 ROS domain)")

    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = Node(
        NODE_NAME, enable_rosout=False, start_parameter_services=False
    )
    position = None
    pose_received = None
    publisher = None
    nonzero_sent = 0

    def on_pose(message: PoseStamped) -> None:
        nonlocal position, pose_received
        p = message.pose.position
        position = (float(p.x), float(p.y), float(p.z))
        pose_received = time.monotonic()

    node.create_subscription(PoseStamped, POSE_TOPIC, on_pose, qos_profile_sensor_data)

    def check(*, own: bool, origin=None) -> None:
        pubs = node.get_publishers_info_by_topic(MOTION_TOPIC)
        subs = node.get_subscriptions_info_by_topic(MOTION_TOPIC)
        qualify_graph(pubs, subs, own_publisher=own)
        qualify_pose(position, receipt_monotonic=pose_received,
                     now=time.monotonic(), origin=origin)

    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if position is not None:
                break
        check(own=False)

        if not execute:
            print("PASS: read-only graph/pose preflight. 0 robot commands; "
                  "gripper disabled. Physical motion NOT tested.")
            return 0

        print("Physical command interface will issue ONE brief X-axis pulse.")
        print("Confirm workcell clear, E-stop in reach, SpaceMouse publisher OFF.")
        # No console auto-arming, no noninteractive operation.
        try:
            answer = input("Type MOVE once to issue the supervised pulse: ")
        except EOFError as exc:
            raise Blocked("Interactive operator confirmation required") from exc
        if answer != "MOVE":
            raise Blocked("Operator did not authorize pulse")
        check(own=False)
        initial = tuple(position)
        publisher = node.create_publisher(Twist, MOTION_TOPIC, 10)
        # Allow discovery; recheck because a competing publisher must abort.
        for _ in range(8):
            rclpy.spin_once(node, timeout_sec=0.05)
        check(own=True, origin=initial)
        if publisher.topic_name != MOTION_TOPIC:
            raise Blocked("Publisher remapped unexpectedly")

        pulse = Twist()
        pulse.linear.x = float(request)
        neutral = Twist()
        begin = time.monotonic()
        next_send = begin
        while time.monotonic() - begin < duration:
            rclpy.spin_once(node, timeout_sec=0.005)
            check(own=True, origin=initial)
            if time.monotonic() >= next_send:
                publisher.publish(pulse)
                nonzero_sent += 1
                next_send += 0.05
        # A neutral message requests target hold; does NOT e-stop robot.
        publisher.publish(neutral)
        print(f"Pulse complete; nonzero requests sent: {nonzero_sent}. "
              "No gripper commands.")
        return 0
    finally:
        if publisher is not None:
            # Do not silently resume nonzero requests even on exceptions.
            try:
                publisher.publish(Twist())
                for _ in range(4):
                    rclpy.spin_once(node, timeout_sec=0.05)
                    publisher.publish(Twist())
            finally:
                node.destroy_publisher(publisher)
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="One interactive, supervised physical X-axis test pulse")
    parser.add_argument("--request", type=float, default=DEFAULT_REQUEST)
    parser.add_argument("--duration", type=float, default=0.10)
    args = parser.parse_args()
    try:
        return run(execute=args.execute, request=args.request, duration=args.duration)
    except Blocked as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
