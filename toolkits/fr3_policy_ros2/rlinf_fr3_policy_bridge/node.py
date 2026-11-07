# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""ROS 2 isolated shadow transport. Never publishes to FR3 motion/gripper topics.

This process refuses the live FR3 ROS domains and any remap of its shadow
Twist publisher. Physical execution needs an independent approved controller
adapter, not a parameter/flag on this node.
"""

from __future__ import annotations

import json
import os
import time

from .gate import OWNER, SYNTHETIC_PROFILE, ShadowGate, ShadowRejected

INPUT = "/rlinf/fr3/shadow/policy_request_json"
HEARTBEAT = "/rlinf/fr3/shadow/heartbeat"
PREVIEW = "/rlinf/fr3/shadow/normalized_twist_preview"
STATUS = "/rlinf/fr3/shadow/status"


def assert_isolated_environment(env: dict[str, str]) -> int:
    if env.get("RLINF_FR3_SHADOW_ONLY") != "1":
        raise PermissionError("Set RLINF_FR3_SHADOW_ONLY=1 to use isolated shadow transport")
    value = env.get("ROS_DOMAIN_ID")
    if value is None or not value.isdecimal():
        raise PermissionError("An explicit isolated ROS_DOMAIN_ID is required")
    domain = int(value)
    if domain in (70, 71) or not 0 <= domain <= 232:
        raise PermissionError("Refusing live FR3/Robotiq ROS domains or invalid domain")
    return domain


def main() -> None:
    assert_isolated_environment(dict(os.environ))
    # Deliberately only imports ROS after verifying isolated-domain constraints.
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String

    class ShadowBridge(Node):
        def __init__(self) -> None:
            super().__init__("rlinf_fr3_policy_shadow")
            self.gate = ShadowGate(
                SYNTHETIC_PROFILE, start_ns=time.monotonic_ns(),
                lease_ns=120_000_000_000,
            )
            self.preview_pub = self.create_publisher(Twist, PREVIEW, 10)
            self.status_pub = self.create_publisher(String, STATUS, 10)
            self.request_sub = self.create_subscription(
                String, INPUT, self.on_request, 10,
            )
            self.heartbeat_sub = self.create_subscription(
                String, HEARTBEAT, self.on_heartbeat, 10,
            )
            # ROS remapping must not silently redirect preview to motor topic.
            if self.preview_pub.topic_name != PREVIEW:
                raise PermissionError("Shadow Twist publisher remapped: refusal")
            self.check_timer = self.create_timer(0.02, self.on_timer)
            self.get_logger().info(
                "FR3 SHADOW ONLY: isolated ROS domain, no robot commands"
            )

        def report(self, obj: dict) -> None:
            out = String()
            out.data = json.dumps(obj, allow_nan=False)
            self.status_pub.publish(out)

        def reject(self, exception: Exception) -> None:
            self.report({"status": "rejected", "reason": str(exception),
                         "state": self.gate.state, "robot_commands_sent": 0})

        def on_heartbeat(self, message: String) -> None:
            try:
                self.gate.heartbeat(message.data, now_ns=time.monotonic_ns())
                self.report({"status": "shadow_heartbeat_ok",
                             "state": self.gate.state, "robot_commands_sent": 0})
            except ShadowRejected as exc:
                self.reject(exc)

        def on_request(self, message: String) -> None:
            try:
                record = json.loads(message.data)
                result = self.gate.accept(record, now_ns=time.monotonic_ns())
                axes = result["shadow_axes"]
                twist = Twist()
                twist.linear.x, twist.linear.y, twist.linear.z = axes[:3]
                twist.angular.x, twist.angular.y, twist.angular.z = axes[3:]
                self.preview_pub.publish(twist)
                self.report({"status": "shadow_accepted", **result})
            except (ValueError, TypeError, ShadowRejected) as exc:
                # Malformed JSON must not leave a usable shadow session.
                self.gate.disable()
                self.reject(exc)

        def on_timer(self) -> None:
            if self.gate.state != "armed_shadow_only":
                return
            try:
                self.gate.check_timeout(now_ns=time.monotonic_ns())
            except ShadowRejected as exc:
                self.reject(exc)

    rclpy.init()
    node = None
    try:
        node = ShadowBridge()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.gate.disable()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
