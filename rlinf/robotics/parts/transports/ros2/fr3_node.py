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

"""System-Python ROS 2 peer for the native FR3 request arm.

Only attaches to existing publishers, controller-manager services and bridges.
Run this file directly so the ROS interpreter need not import ML dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

from fr3_runtime import (
    ARM_TOPIC,
    GRIP_JOINT,
    GRIP_TOPIC,
    JOINTS,
    STATE_PREFIX,
    RequestWindow,
    finite_vector,
)


class RosPeer:
    """Own two DDS contexts and one optional pair of command publishers."""

    def __init__(self, session: Path, log: Path) -> None:
        import rclpy
        from controller_manager_msgs.srv import ListControllers
        from franka_msgs.msg import FrankaRobotState
        from geometry_msgs.msg import PoseStamped, WrenchStamped
        from rcl_interfaces.msg import Log
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState

        self.rclpy = rclpy
        self.session = session
        self.log = log.open("x", buffering=1)
        owner = json.loads((session / "owner.json").read_text())
        self.window = RequestWindow(owner.get("duration_s", 3))
        self.cache = {}
        self.pubs = None
        self.graph_ok = False
        self.controller_ok = False
        self.controller_service_ready = False
        self.controller_at = -math.inf
        self.next_graph = 0.0
        self.future = None
        self.origin = None
        self.graph_details = {}
        self.controller_details = []
        self.health_failures = {}
        self._health_signature = None
        self._last_health_log = -math.inf
        self.contexts, self.nodes, self.executors = [], [], []
        for domain in (70, 71):
            context = Context()
            rclpy.init(context=context, domain_id=domain)
            node = rclpy.create_node(
                f"rlinf_fr3_request_{os.getpid()}_{domain}", context=context
            )
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            self.contexts.append(context)
            self.nodes.append(node)
            self.executors.append(executor)
        arm, grip = self.nodes
        self.subs = [
            arm.create_subscription(
                PoseStamped,
                STATE_PREFIX + "/current_pose",
                lambda m: self.receive("pose", m),
                qos_profile_sensor_data,
            ),
            arm.create_subscription(
                JointState,
                STATE_PREFIX + "/measured_joint_states",
                lambda m: self.receive("joints", m),
                qos_profile_sensor_data,
            ),
            arm.create_subscription(
                WrenchStamped,
                STATE_PREFIX + "/external_wrench_in_base_frame",
                lambda m: self.receive("wrench", m),
                qos_profile_sensor_data,
            ),
            arm.create_subscription(
                FrankaRobotState,
                STATE_PREFIX + "/robot_state",
                lambda m: self.receive("robot", m),
                qos_profile_sensor_data,
            ),
            grip.create_subscription(
                JointState,
                "/joint_states",
                lambda m: self.receive("gripper", m),
                qos_profile_sensor_data,
            ),
            arm.create_subscription(Log, "/rosout", self.diagnostic, 100),
        ]
        self.client = arm.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        self.request_type = ListControllers.Request
        self.parent_pid = os.getppid()
        self.driver_pid = owner["pid"]
        self.event("attached", domains=[70, 71], command_publishers_created=False)

    def event(self, kind: str, **values: object) -> None:
        """Append host-monotonic evidence without blocking on ROS logging."""
        self.log.write(
            json.dumps(
                {"event": kind, "monotonic": time.monotonic(), **values},
                allow_nan=False,
            )
            + "\n"
        )

    def diagnostic(self, msg: object) -> None:
        """Retain actual controller diagnostic text for operator review."""
        if msg.name == "joint_impedance_ik_controller" and "FR3_DIAG_V1" in msg.msg:
            self.event("controller_diagnostic", text=msg.msg)

    def receive(self, key: str, msg: object) -> None:
        """Refresh only advancing source stamps and the physical gripper joint."""
        if key == "gripper" and GRIP_JOINT not in msg.name:
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        prior = self.cache.get(key)
        if stamp == (0, 0) or (prior and stamp <= prior[2]):
            return
        self.cache[key] = (msg, time.monotonic(), stamp)

    def spin(self) -> None:
        """Service bounded ROS work and refresh controller ownership evidence."""
        for executor in self.executors:
            for _ in range(8):
                executor.spin_once(timeout_sec=0)
        now = time.monotonic()
        if self.future is not None and self.future.done():
            result = self.future.result()
            controllers = [] if result is None else result.controller
            self.controller_details = [
                {
                    "name": c.name,
                    "state": c.state,
                    "claimed_interfaces": list(c.claimed_interfaces),
                }
                for c in controllers
            ]
            active = [
                c for c in controllers if c.state == "active" and c.claimed_interfaces
            ]
            self.controller_ok = (
                len(active) == 1
                and active[0].name == "joint_impedance_ik_controller"
                and set(active[0].claimed_interfaces) == {j + "/effort" for j in JOINTS}
            )
            self.controller_at = now
            self.future = None
        if now >= self.next_graph:
            arm = self.nodes[0]
            expected = 1 if self.pubs is not None else 0
            self.graph_details = {
                topic: {
                    "publishers": arm.count_publishers(topic),
                    "subscribers": arm.count_subscribers(topic),
                }
                for topic in (ARM_TOPIC, GRIP_TOPIC)
            }
            self.graph_ok = all(
                counts["publishers"] == expected and counts["subscribers"] == 1
                for counts in self.graph_details.values()
            )
            self.next_graph = now + 0.2
            self.controller_service_ready = self.client.service_is_ready()
            if self.future is None and self.controller_service_ready:
                self.future = self.client.call_async(self.request_type())

    def observation(self) -> dict:
        """Return the training-order 19D state, measured in the recorded frames."""
        now = time.monotonic()
        required = ("pose", "joints", "wrench", "robot", "gripper")
        unavailable = {
            k: {"age_s": now - self.cache[k][1]} if k in self.cache else "missing"
            for k in required
            if k not in self.cache or now - self.cache[k][1] > 0.25
        }
        if unavailable:
            raise RuntimeError(
                "Missing or stale ROS state (250 ms maximum receipt age): "
                + json.dumps(unavailable)
            )
        pose, joints, wrench, robot, grip = [self.cache[k][0] for k in required]
        if pose.header.frame_id != "base" or wrench.header.frame_id != "base":
            raise RuntimeError(
                "Expected measured pose and wrench in base frame; found "
                f"pose={pose.header.frame_id!r}, wrench={wrench.header.frame_id!r}"
            )
        active_errors = [
            name
            for name in robot.current_errors.get_fields_and_field_types()
            if getattr(robot.current_errors, name)
        ]
        if robot.robot_mode != 2 or active_errors:
            raise RuntimeError(
                "FR3 is not in healthy mode 2 or reports a current error; "
                f"robot_mode={robot.robot_mode}, active_errors={active_errors}"
            )
        p, q, f, t = (
            pose.pose.position,
            pose.pose.orientation,
            wrench.wrench.force,
            wrench.wrench.torque,
        )
        qv = finite_vector([q.x, q.y, q.z, q.w], 4)
        if not 0.95 <= math.sqrt(sum(x * x for x in qv)) <= 1.05:
            raise RuntimeError("Measured quaternion is not approximately unit length")
        try:
            joint_values = [joints.position[joints.name.index(j)] for j in JOINTS]
            knuckle = grip.position[grip.name.index(GRIP_JOINT)]
        except (ValueError, IndexError) as exc:
            raise RuntimeError("Missing named physical joints") from exc
        torque = finite_vector([t.x, t.y, t.z], 3)
        state = finite_vector(
            [
                p.x,
                p.y,
                p.z,
                *qv,
                *joint_values,
                f.x,
                f.y,
                f.z,
                knuckle,
                math.sqrt(sum(x * x for x in torque)),
            ],
            19,
        )
        if self.origin is not None and math.dist(state[:3], self.origin) > 0.08:
            raise RuntimeError("Measured translation exceeded the 8 cm episode bound")
        return {
            "state": state,
            "received": min(self.cache[k][1] for k in required),
            "source_stamps": {k: self.cache[k][2] for k in required},
            "stopped": self.window.reason,
        }

    def healthy(self) -> bool:
        """Require fresh state, a live driver and exclusive active controller."""
        failures = {}
        try:
            self.observation()
        except (RuntimeError, ProcessLookupError, PermissionError) as exc:
            failures["observation"] = {
                "exception": type(exc).__name__,
                "message": str(exc),
            }
        except Exception as exc:
            self.event(
                "health_exception",
                condition="observation",
                exception=type(exc).__name__,
                message=str(exc),
            )
            raise
        try:
            os.kill(self.driver_pid, 0)
        except (RuntimeError, ProcessLookupError, PermissionError) as exc:
            failures["driver_liveness"] = {
                "pid": self.driver_pid,
                "exception": type(exc).__name__,
                "message": str(exc),
            }
        now = time.monotonic()
        if os.getppid() != self.parent_pid:
            failures["parent_identity"] = {
                "expected_pid": self.parent_pid,
                "actual_pid": os.getppid(),
            }
        if not self.graph_ok:
            failures["command_graph"] = {
                "expected_publishers": 1 if self.pubs is not None else 0,
                "expected_subscribers": 1,
                "topics": self.graph_details,
            }
        if not self.controller_ok:
            failures["controller_ownership"] = {
                "controllers": self.controller_details,
                "service_ready": self.controller_service_ready,
                "request_pending": self.future is not None,
            }
        controller_age = now - self.controller_at
        if not controller_age < 1.0:
            failures["controller_response_freshness"] = {
                "age_s": controller_age if math.isfinite(controller_age) else None,
                "max_age_s": 1.0,
            }
        if (self.session / "STOP").exists():
            failures["operator_stop"] = str(self.session / "STOP")
        if self.window.started is not None and not (self.session / "ARMED").exists():
            failures["operator_permission"] = str(self.session / "ARMED")
        self.health_failures = failures
        signature = tuple(failures)
        if signature != self._health_signature or (
            failures and now - self._last_health_log >= 1.0
        ):
            self.event(
                "health_check",
                healthy=not failures,
                failed_conditions=failures,
                state_receipts={
                    key: {"age_s": now - reading[1], "source_stamp": reading[2]}
                    for key, reading in self.cache.items()
                },
            )
            self._health_signature = signature
            self._last_health_log = now
        return not failures

    def command(self, command: dict) -> dict:
        """Handle an observation, expiring sample, or explicit stop."""
        op = command["op"]
        if op == "observe":
            if not self.healthy():
                raise RuntimeError(
                    "ROS state, existing controller, bridge, or exclusive ownership unavailable: "
                    + json.dumps(self.health_failures)
                )
            observation = self.observation()
            self.event("observed_state", **observation)
            return observation
        if op == "stop":
            self.window.stop("operator_or_client_stop")
            return {"stopped": self.window.reason}
        if op == "reserve":
            if self.pubs is not None or not self.healthy():
                raise RuntimeError("Cannot reserve the command topics")
            from geometry_msgs.msg import Twist
            from std_msgs.msg import Float32

            self.pubs = (
                self.nodes[0].create_publisher(Twist, ARM_TOPIC, 1),
                self.nodes[0].create_publisher(Float32, GRIP_TOPIC, 1),
            )
            self.graph_ok = False
            self.next_graph = 0
            return {"reserved": True, "commands_published": 0}
        if op != "request":
            raise ValueError(f"Unknown ROS peer operation: {op}")
        if self.window.reason:
            return {
                "stopped": self.window.reason,
                "deadline": self.window.deadline,
            }
        if not (self.session / "ARMED").is_file() or not self.healthy():
            self.window.stop("operator_or_runtime_not_ready")
            raise RuntimeError(
                "Operator confirmation or runtime safety condition missing"
            )
        if self.pubs is None:
            raise RuntimeError("Command endpoints have not been reserved")
        if self.origin is None:
            self.origin = self.observation()["state"][:3]
        now = time.monotonic()
        edge = self.window.submit(command["value"], now, command["issued"])
        if edge is not None and not self.window.reason:
            from std_msgs.msg import Float32

            self.pubs[1].publish(Float32(data=edge))
            self.event("gripper_request", value=edge)
        self.event("policy_request", value=command["value"], issued=command["issued"])
        return {"stopped": self.window.reason, "deadline": self.window.deadline}

    def publish(self) -> None:
        """Refresh unchanged Twist at 100 Hz, with local deadline enforcement."""
        now = time.monotonic()
        value = self.window.tick(now, self.healthy())
        if self.pubs is None or self.window.started is None:
            return
        from geometry_msgs.msg import Twist

        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = value[:3]
        msg.angular.x, msg.angular.y, msg.angular.z = value[3:]
        self.pubs[0].publish(msg)
        self.event("twist_refresh", value=value, stopped=self.window.reason)

    def close(self) -> None:
        """Send a bounded neutral tail and destroy only this peer's resources."""
        self.window.stop("peer_exit")
        for _ in range(10):
            self.publish()
            time.sleep(0.01)
        for executor, node, context in zip(self.executors, self.nodes, self.contexts):
            executor.shutdown(timeout_sec=0)
            node.destroy_node()
            self.rclpy.shutdown(context=context)
        self.log.close()


def main() -> None:
    """Serve newline-delimited JSON on stdin/stdout until the parent closes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    incoming = queue.Queue(maxsize=2)

    def read() -> None:
        try:
            for line in sys.stdin:
                incoming.put(json.loads(line))
        finally:
            incoming.put(None)

    threading.Thread(target=read, daemon=True).start()
    peer = RosPeer(args.session, args.log)
    next_tick = time.monotonic()
    try:
        while os.getppid() == peer.parent_pid:
            peer.spin()
            now = time.monotonic()
            if now >= next_tick:
                peer.publish()
                next_tick += 0.01
                if next_tick < now:
                    next_tick = now + 0.01
            try:
                command = incoming.get_nowait()
            except queue.Empty:
                time.sleep(0.0005)
                continue
            if command is None:
                break
            try:
                response = {"ok": True, **peer.command(command)}
            except Exception as exc:
                if command.get("op") != "observe":
                    peer.window.stop(str(exc))
                response = {"ok": False, "error": str(exc)}
            sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
            sys.stdout.flush()
    finally:
        peer.close()


if __name__ == "__main__":
    main()
