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
import json
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
MAX_PULSE_S = 0.50  # Longer low-amplitude pulse to distinguish motion from pose noise
POSE_MAX_AGE_S = 0.2
MAX_OBSERVED_DISPLACEMENT_M = 0.01
POST_PULSE_OBSERVE_S = 0.5
MIN_DETECTABLE_X_M = 0.0002  # diagnostic threshold, not a safety threshold
EXPECTED_TYPE = "geometry_msgs/msg/Twist"


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
    # The workstation's Jazzy graph reports Humble/Nuc endpoints with
    # _NODE_NAME_UNKNOWN_ even when a separately verified active controller
    # is subscribed. Require exactly one matching-Twist subscriber instead
    # of treating a DDS node-name reporting limitation as missing control.
    if len(subscribers) != 1:
        raise Blocked(
            f"Expected one FR3 arm subscriber, found {len(subscribers)}"
        )
    peer = subscribers[0]
    name = peer.node_name
    topic_type = getattr(peer, "topic_type", None)
    if name not in (CONTROLLER, "_NODE_NAME_UNKNOWN_"):
        raise Blocked(f"Unexpected motion subscriber node: {name}")
    if topic_type is not None and topic_type != EXPECTED_TYPE:
        raise Blocked(f"Unexpected motion subscriber type: {topic_type}")
    if name == "_NODE_NAME_UNKNOWN_" and topic_type != EXPECTED_TYPE:
        raise Blocked("Unnamed subscriber requires verified Twist type")


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
        raise Blocked("Only a single <=0.05 normalized X request, <=0.50s pulse is allowed")


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
    max_travel_m = 0.0

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
        deadline = time.monotonic() + 10.0
        last_reason = "ROS discovery incomplete"
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                check(own=False)
                break
            except Blocked as exc:
                last_reason = str(exc)
                # A competing command publisher is never treated as a
                # transient discovery failure.
                if "Existing command publisher" in last_reason:
                    raise
        else:
            raise Blocked(f"ROS graph/pose preflight timed out: {last_reason}")

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
        # input() blocks the single-threaded ROS executor, so the pose
        # received before the operator prompt is necessarily stale.
        # Resume spinning, require a NEW measured pose after confirmation,
        # and keep checking for any competing command publisher.
        confirmation_at = time.monotonic()
        refresh_deadline = confirmation_at + 3.0
        while time.monotonic() < refresh_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            qualify_graph(
                node.get_publishers_info_by_topic(MOTION_TOPIC),
                node.get_subscriptions_info_by_topic(MOTION_TOPIC),
                own_publisher=False,
            )
            if pose_received is not None and pose_received > confirmation_at:
                qualify_pose(
                    position, receipt_monotonic=pose_received,
                    now=time.monotonic(),
                )
                break
        else:
            raise Blocked(
                "No fresh FR3 pose received after operator confirmation; "
                "no motion command published"
            )
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
            max_travel_m = max(max_travel_m, math.dist(initial, position))
            if time.monotonic() >= next_send:
                publisher.publish(pulse)
                nonzero_sent += 1
                next_send += 0.05
        # A neutral message requests target hold; does NOT e-stop robot.
        publisher.publish(neutral)
        # Observe residual motion with neutral commands and the same watchdog.
        end_observe = time.monotonic() + POST_PULSE_OBSERVE_S
        while time.monotonic() < end_observe:
            rclpy.spin_once(node, timeout_sec=0.02)
            check(own=True, origin=initial)
            publisher.publish(neutral)
            max_travel_m = max(max_travel_m, math.dist(initial, position))
        delta = tuple(b - a for a, b in zip(initial, position))
        x_detected = delta[0] * (1 if request > 0 else -1) >= MIN_DETECTABLE_X_M
        result = {
            "mode": "single_supervised_arm_pulse",
            "request_6d": [request, 0., 0., 0., 0., 0.],
            "pulse_s": duration,
            "nonzero_ros_publishes": nonzero_sent,
            "measured_tcp_delta_mm": [round(d * 1000, 3) for d in delta],
            "max_measured_tcp_displacement_mm": round(max_travel_m * 1000, 3),
            "requested_x_direction_observed": bool(x_detected),
            "identity_policy_mapping_approved": False,
            "note": "One brief X pulse cannot verify all six axes, time alignment or controller acceptance.",
        }
        print(json.dumps(result, indent=2))
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
