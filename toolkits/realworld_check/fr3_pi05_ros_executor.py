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
        GripperRequestFilter,
        identity_arm_request,
        load_mapping_verification,
        validate_action_chunk,
    )
else:
    from fr3_pi05_request_adapter import (
        GripperRequestFilter,
        identity_arm_request,
        load_mapping_verification,
        validate_action_chunk,
    )

ARM_TOPIC = "/franka_controller/target_cartesian_velocity_percent"
GRIPPER_TOPIC = "/gripper_client/target_gripper_width_percent"
POSE_TOPIC = "/franka_robot_state_broadcaster/current_pose"
NAME = "rlinf_fr3_pi05_executor"
PERIOD_S = 0.1  # SFT policy labels at 10 Hz
PUBLISH_PERIOD_NS = 10_000_000  # transport refresh, never another policy step
MAX_PUBLISH_GAP_NS = 50_000_000  # fail closed after a scheduler stall
LEGACY_GAIN = 0.15  # existing experimental motion limit; NOT trained mapping
POSE_TIMEOUT_S = 0.25
MAX_EPISODE_S = 20.0
MAX_TRAVEL_M = 0.08


def validate_chunk(
    chunk, *, observation_ns: int, issued_ns: int, now_ns: int
) -> list[list[float]]:
    if not isinstance(chunk, list) or len(chunk) != 10:
        raise ValueError("Expected exactly ten 7D requests")
    if (
        not all(type(t) is int for t in (observation_ns, issued_ns))
        or not 0 < observation_ns <= issued_ns <= now_ns
        or now_ns - observation_ns > 1_000_000_000
        or now_ns - issued_ns > 300_000_000
    ):
        raise ValueError("Stale observation or issued request")
    return validate_action_chunk(chunk)


class CommandSchedule:
    """Hold each validated policy sample for 100 ms, then expire to neutral.

    Admission checks observation/issue age; a chunk has exactly one second
    of execution authority. A delayed scheduler faults instead of replaying
    missed samples or extending a sample's deadline.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Revoke the active chunk and its timing state."""
        self.batch = []
        self.started_ns = None
        self.last_tick_ns = None

    def start(
        self,
        values: list[list[float]],
        *,
        observation_ns: int,
        issued_ns: int,
        now_ns: int,
    ) -> None:
        """Admit ten fresh 7D requests when no chunk is active."""
        if self.batch:
            raise RuntimeError("Previous chunk not finished")
        self.batch = validate_chunk(
            values, observation_ns=observation_ns, issued_ns=issued_ns, now_ns=now_ns
        )
        self.started_ns = now_ns
        self.last_tick_ns = now_ns

    def sample(self, now_ns: int) -> tuple[int, list[float]] | None:
        """Return the current sample, or neutral when the chunk has expired."""
        if not self.batch:
            return None
        if not 0 <= now_ns - self.last_tick_ns <= MAX_PUBLISH_GAP_NS:
            self.clear()
            raise RuntimeError("Command scheduler missed its 50 ms deadline")
        self.last_tick_ns = now_ns
        index = (now_ns - self.started_ns) // int(PERIOD_S * 1_000_000_000)
        if index >= len(self.batch):
            self.clear()
            return None
        return int(index), self.batch[index]


def main() -> int:
    if os.environ.get("ROS_DOMAIN_ID") != "70":
        raise RuntimeError("This executor requires the explicit FR3 domain 70")
    mapping_mode = os.environ.get("FR3_PI05_MAPPING_MODE", "legacy_limited")
    if mapping_mode not in ("identity_verified", "legacy_limited"):
        raise ValueError("Unknown FR3 request mapping mode")
    verification = None
    if mapping_mode == "identity_verified":
        verification = load_mapping_verification(
            os.environ.get("FR3_PI05_MAPPING_VERIFICATION")
        )
        if (
            os.environ.get("FR3_PI05_MAPPING_VERIFICATION_SHA256")
            != verification["record_sha256"]
        ):
            raise ValueError(
                "Mapping verification changed or parent approval digest missing"
            )
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float32

    rclpy.init()
    node = Node(NAME, enable_rosout=False, start_parameter_services=False)
    pose = None
    pose_at = 0.0
    origin = None
    started = 0.0
    schedule = CommandSchedule()
    last_action = None
    next_publish_ns = 0
    publisher = None
    gripper_pub = None
    total_commands = 0
    arm_publications = 0
    last_publish_ns = None
    max_publish_gap_ns = 0
    episode_ended = False
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
        if (
            pose is None
            or not all(map(math.isfinite, pose))
            or time.monotonic() - pose_at > POSE_TIMEOUT_S
        ):
            raise RuntimeError("FR3 pose missing or stale")
        if origin is not None and math.dist(origin, pose) > MAX_TRAVEL_M:
            raise RuntimeError("Measured FR3 displacement exceeded 8 cm limit")
        if origin is not None and time.monotonic() - started > MAX_EPISODE_S:
            raise RuntimeError("Supervised episode time limit reached")

    def exclusive():
        publishers = node.get_publishers_info_by_topic(ARM_TOPIC)
        if len(publishers) != 1 or publishers[0].node_name != NAME:
            raise RuntimeError(
                "SpaceMouse/other publisher detected: exclusive control lost"
            )
        if not node.get_subscriptions_info_by_topic(ARM_TOPIC):
            raise RuntimeError("No FR3 controller subscriber")
        if gripper_pub is not None:
            writers = node.get_publishers_info_by_topic(GRIPPER_TOPIC)
            if len(writers) != 1 or writers[0].node_name != NAME:
                raise RuntimeError("Competing gripper publisher detected")
            if not node.get_subscriptions_info_by_topic(GRIPPER_TOPIC):
                raise RuntimeError("Robotiq bridge subscriber missing")

    def publish_arm(msg, *, expires_ns=None):
        nonlocal arm_publications, last_publish_ns, max_publish_gap_ns
        if expires_ns is not None:
            now_ns = time.monotonic_ns()
            if now_ns >= expires_ns:
                raise RuntimeError("Policy sample expired before publication")
            if (
                last_publish_ns is not None
                and now_ns - last_publish_ns > MAX_PUBLISH_GAP_NS
            ):
                raise RuntimeError("Arm publication missed its 50 ms deadline")
        publisher.publish(msg)
        now_ns = time.monotonic_ns()
        if last_publish_ns is not None:
            max_publish_gap_ns = max(max_publish_gap_ns, now_ns - last_publish_ns)
        last_publish_ns = now_ns
        arm_publications += 1

    def neutral():
        if publisher is not None:
            publish_arm(Twist())

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
                raise RuntimeError(
                    "Stop SpaceMouse arm publisher before policy rollout"
                )
            if node.get_publishers_info_by_topic(GRIPPER_TOPIC):
                raise RuntimeError(
                    "Stop SpaceMouse gripper publisher before policy rollout"
                )
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
        publisher = node.create_publisher(Twist, ARM_TOPIC, 1)
        gripper_pub = node.create_publisher(Float32, GRIPPER_TOPIC, 10)
        # ROS graph propagation.
        for _ in range(12):
            rclpy.spin_once(node, timeout_sec=0.05)
        exclusive()
        emit(
            event="ready",
            arm_topic=ARM_TOPIC,
            gripper_topic=GRIPPER_TOPIC,
            mapping_mode=mapping_mode,
            arm_gain=gain,
            policy_period_s=PERIOD_S,
            publish_period_ns=PUBLISH_PERIOD_NS,
            mapping_verification_sha256=verification["record_sha256"]
            if verification
            else None,
        )

        input_buffer = b""
        quitting = False
        while not quitting:
            rclpy.spin_once(node, timeout_sec=0.002)
            # Read available bytes only: a partial JSON line must not block
            # command expiry, neutral refresh, or the measured-pose watchdog.
            if select.select([sys.stdin], [], [], 0)[0]:
                data = os.read(sys.stdin.fileno(), 65536)
                if not data:  # parent died: never continue policy control
                    break
                input_buffer += data
                if len(input_buffer) > 65536:
                    raise RuntimeError("Executor input exceeded 64 KiB bound")
            while b"\n" in input_buffer:
                line, input_buffer = input_buffer.split(b"\n", 1)
                item = json.loads(line)
                op = item.get("op")
                if op == "begin_episode":
                    if origin is not None:
                        raise RuntimeError("Episode already active")
                    if (
                        verification
                        and item.get("mapping_verification_sha256")
                        != verification["record_sha256"]
                    ):
                        raise RuntimeError(
                            "Episode requires explicit approval of the verified mapping record"
                        )
                    fresh_pose()
                    origin = tuple(pose)
                    episode_ended = False
                    started = time.monotonic()
                    gripper_filter.reset()
                    total_commands = 0
                    arm_publications = 0
                    last_publish_ns = None
                    max_publish_gap_ns = 0
                    next_publish_ns = time.monotonic_ns()
                    emit(event="episode_started", tcp_xyz=origin)
                elif op == "chunk":
                    if origin is None:
                        raise RuntimeError("No active episode")
                    schedule.start(
                        item.get("values"),
                        observation_ns=item.get("observation_ns"),
                        issued_ns=item.get("issued_ns"),
                        now_ns=time.monotonic_ns(),
                    )
                    last_action = None
                    next_publish_ns = time.monotonic_ns()
                elif op == "end_episode":
                    if episode_ended:
                        continue
                    schedule.clear()
                    origin = None
                    neutral()
                    episode_ended = True
                    emit(
                        event="episode_ended",
                        commands_sent=total_commands,
                        arm_publications=arm_publications,
                        max_publish_gap_ns=max_publish_gap_ns,
                    )
                elif op == "quit":
                    schedule.clear()
                    quitting = True
                    break
                else:
                    raise RuntimeError("Unexpected executor operation")
            if quitting:
                break
            fresh_pose()
            exclusive()
            now_ns = time.monotonic_ns()
            had_batch = bool(schedule.batch)
            sample = schedule.sample(now_ns)
            if had_batch and sample is None:
                neutral()
                next_publish_ns = now_ns + PUBLISH_PERIOD_NS
                emit(event="chunk_done", commands_sent=total_commands)
            if origin is not None and now_ns >= next_publish_ns:
                # Never catch up a delayed transport timer with a publish burst.
                next_publish_ns += (
                    (now_ns - next_publish_ns) // PUBLISH_PERIOD_NS + 1
                ) * PUBLISH_PERIOD_NS
                if sample is None:
                    neutral()
                else:
                    index, values = sample
                    # Raw normalized SpaceMouse request axis order (not SI velocity).
                    arm = identity_arm_request(values)
                    sent_arm = (
                        list(arm)
                        if mapping_mode == "identity_verified"
                        else [x * LEGACY_GAIN for x in arm]
                    )
                    msg = Twist()
                    msg.linear.x, msg.linear.y, msg.linear.z = sent_arm[:3]
                    msg.angular.x, msg.angular.y, msg.angular.z = sent_arm[3:]
                    publish_arm(
                        msg,
                        expires_ns=schedule.started_ns
                        + (index + 1) * int(PERIOD_S * 1_000_000_000),
                    )
                    if index == last_action:
                        continue
                    last_action = index
                    total_commands += 1  # policy samples, not ROS refreshes
                    g = values[6]
                    gripper_event = gripper_filter.step(g, now=time.monotonic())
                    if gripper_event != "hold":
                        gripper_msg = Float32()
                        gripper_msg.data = 0.0 if gripper_event == "close" else 1.0
                        gripper_pub.publish(gripper_msg)
                    emit(
                        event="action",
                        command_index=index,
                        requested_7d=values,
                        sent_arm_6d=sent_arm,
                        gripper_event=gripper_event,
                        tcp_xyz=pose,
                        monotonic_ns=time.monotonic_ns(),
                        commands_sent=total_commands,
                    )
        return 0
    except Exception as exc:
        schedule.clear()
        neutral()
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
