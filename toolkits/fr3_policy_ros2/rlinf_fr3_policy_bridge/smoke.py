"""Exercise full ROS 2 transport in isolated domain without robot topics."""

import json
import os
import time

from .gate import OWNER, make_synthetic_request
from .node import HEARTBEAT, INPUT, PREVIEW, STATUS, assert_isolated_environment


def main() -> None:
    assert_isolated_environment(dict(os.environ))
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("rlinf_fr3_shadow_smoke_client")
    previews = []
    reports = []
    pub_heartbeat = node.create_publisher(String, HEARTBEAT, 10)
    pub_request = node.create_publisher(String, INPUT, 10)
    s1 = node.create_subscription(Twist, PREVIEW, previews.append, 10)
    s2 = node.create_subscription(String, STATUS, reports.append, 10)
    try:
        # Allow DDS discovery on the isolated domain, fail if peer absent.
        deadline = time.monotonic() + 7.0
        while time.monotonic() < deadline:
            if pub_request.get_subscription_count() > 0 and pub_heartbeat.get_subscription_count() > 0:
                break
            rclpy.spin_once(node, timeout_sec=0.05)
        else:
            raise RuntimeError("No isolated shadow bridge subscriptions discovered")
        hb = String()
        hb.data = OWNER
        pub_heartbeat.publish(hb)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            if any(json.loads(r.data).get("status") == "shadow_heartbeat_ok"
                   for r in reports):
                break
        else:
            raise RuntimeError("Shadow heartbeat acknowledgment not received")
        now_ns = time.monotonic_ns()
        request = String()
        request.data = json.dumps(make_synthetic_request(
            sequence=1, observed_ns=now_ns-2_000_000,
            issued_ns=now_ns-1_000_000,
        ))
        pub_request.publish(request)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if previews and any(json.loads(r.data).get("status") == "shadow_accepted" for r in reports):
                break
        else:
            raise RuntimeError("No accepted shadow request / preview Twist received: " +
                               str([x.data for x in reports]))
        msg = previews[0]
        if abs(msg.linear.x - (0.001/0.007)) > 1e-9:
            raise AssertionError("Unexpected mathematical preview conversion")
        accepted = next(json.loads(r.data) for r in reports
                        if json.loads(r.data).get("status") == "shadow_accepted")
        if accepted["robot_commands_sent"] != 0:
            raise AssertionError("Unexpected physical command assertion")
        print("PASS: isolated ROS2 shadow transport; normalized preview only; 0 robot commands")
    finally:
        node.destroy_subscription(s1)
        node.destroy_subscription(s2)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
