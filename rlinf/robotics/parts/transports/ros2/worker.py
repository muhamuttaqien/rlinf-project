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

"""Standalone ROS-compatible subscriber process. Run by ROS2ObservationTransport."""

import argparse
import select
import signal
import socket
import time
from typing import Any

from passive import collect_status

# Executed by path so importing this module never imports RLinf or its dependencies.
from protocol import PROFILES, decode_message, receive_packet, send_packet


def serve(channel: socket.socket, profile: str, domain_id: int) -> None:
    """Serve latest-only state snapshots until shutdown or parent EOF."""
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message

    running = True

    def stop(signum: int, frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    context = Context()
    node = executor = None
    try:
        rclpy.init(args=[], context=context, domain_id=domain_id)
        node = Node(
            "rlinf_" + profile + "_observations",
            context=context,
            enable_rosout=False,
            start_parameter_services=False,
            use_global_arguments=False,
        )
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        specs = PROFILES[profile]
        streams = {
            key: {
                "topic": topic,
                "type": ros_type,
                "sample": None,
                "sequence": 0,
                "error": None,
                "graph_error": None,
                "publishers": 0,
            }
            for key, (topic, ros_type, _) in specs.items()
        }
        subscriptions = {}

        def callback(msg: Any, topic: str) -> None:
            receipt = time.monotonic_ns()
            for key, (expected_topic, ros_type, joints) in specs.items():
                if expected_topic != topic:
                    continue
                stream = streams[key]
                try:
                    sample = decode_message(msg, ros_type, joints)
                    if sample is None:
                        continue
                    sample["received_monotonic_ns"] = receipt
                    stream["sample"] = sample
                    stream["sequence"] += 1
                    stream["error"] = None
                except (ValueError, TypeError, AttributeError, OverflowError) as exc:
                    stream["sample"] = None
                    stream["error"] = str(exc)[:1024]

        send_packet(
            channel, {"op": "ready", "domain_id": domain_id, "profile": profile}
        )
        next_discovery = 0.0
        while running and context.ok():
            if time.monotonic() >= next_discovery:
                next_discovery = time.monotonic() + 0.5
                for key, (topic, ros_type, _) in specs.items():
                    infos = node.get_publishers_info_by_topic(topic)
                    stream = streams[key]
                    stream["publishers"] = sum(i.topic_type == ros_type for i in infos)
                    stream["graph_error"] = None
                    if any(i.topic_type != ros_type for i in infos):
                        stream["graph_error"] = (
                            f"Expected {ros_type}; found conflicting publisher types"
                        )
                        stream["sample"] = None
                        continue
                    if stream["publishers"] and topic not in subscriptions:
                        subscriptions[topic] = node.create_subscription(
                            get_message(ros_type),
                            topic,
                            lambda msg, topic=topic: callback(msg, topic),
                            qos,
                        )
            executor.spin_once(timeout_sec=0.01)
            if select.select([channel], [], [], 0)[0]:
                request = receive_packet(channel)
                if request.get("op") == "shutdown":
                    send_packet(channel, {"op": "stopped"})
                    break
                if request.get("op") == "inspect_status":
                    result = collect_status(
                        node, executor, profile, request.get("budget")
                    )
                    send_packet(
                        channel,
                        {
                            "op": "inspect_status",
                            "domain_id": domain_id,
                            "profile": profile,
                            "evidence": result,
                        },
                    )
                    continue
                if request.get("op") != "snapshot":
                    raise ValueError("Expected snapshot or shutdown IPC request")
                send_packet(
                    channel,
                    {
                        "op": "snapshot",
                        "domain_id": domain_id,
                        "profile": profile,
                        "streams": streams,
                    },
                )
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        if node is not None:
            node.destroy_node()
        context.try_shutdown()


def main() -> None:
    """Parse the inherited IPC descriptor and explicit ROS domain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    args = parser.parse_args()
    if not 0 <= args.domain_id <= 232:
        parser.error("Expected ROS domain ID in [0, 232]")
    with socket.socket(fileno=args.fd) as channel:
        # Bound an abandoned peer's send as well as parent-side requests.
        channel.settimeout(2.0)
        try:
            serve(channel, args.profile, args.domain_id)
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as exc:
            send_packet(channel, {"op": "error", "error": str(exc)[:2048]})
            raise


if __name__ == "__main__":
    main()
