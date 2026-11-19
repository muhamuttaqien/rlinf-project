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

"""Local mock endpoint worker: stdlib IPC only, no ROS or hardware imports."""

import argparse
import math
import os
import socket
import sys
import time
from pathlib import Path

# -I/-S exclude environment, cwd and site hooks; expose the sibling protocol only.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import receive_packet, send_packet  # noqa: E402

ENDPOINTS = {
    "fr3_normalized_pose_offset_v1": "mock://fr3/twist",
    "robotiq_normalized_closure_v1": "mock://robotiq/gripper",
}
BEHAVIORS = {
    "normal",
    "drop",
    "reject",
    "fail",
    "preempt",
    "ownership_conflict",
    "identity_mismatch",
    "inconsistent",
    "worker_crash",
    "ipc_failure",
    "hang",
}


def main() -> None:
    """Simulate one outstanding request, advancing logical time only on IPC calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    args = parser.parse_args()
    identity = None
    session = authority = None
    active = None
    wire = None
    behavior = "normal"
    delay_ns = 0
    count = 0
    last_now = 0
    with socket.socket(fileno=args.fd) as channel:
        channel.settimeout(None)
        send_packet(channel, {"op": "ready", "offline": True, "ros_domain_id": None})
        try:
            while True:
                request = receive_packet(channel)
                op = request["op"]
                if op == "shutdown":
                    send_packet(channel, {"op": "stopped", "offline": True})
                    return
                if op == "configure":
                    identity = request["interface"]
                    if identity.get("endpoint") != ENDPOINTS.get(identity.get("kind")):
                        raise ValueError(
                            "Expected a fixed mock endpoint, never a ROS topic/domain"
                        )
                    session, authority = request["session_id"], request["authority_id"]
                    active = None
                elif op == "behavior":
                    if (
                        request["behavior"] not in BEHAVIORS
                        or type(request["delay_ns"]) is not int
                        or request["delay_ns"] < 0
                    ):
                        raise ValueError(
                            "Expected a known mock behavior and nonnegative integer delay"
                        )
                    behavior, delay_ns = request["behavior"], request["delay_ns"]
                elif op in ("status", "submit", "poll", "cancel"):
                    now = request["now_ns"]
                    if type(now) is not int or now < last_now or identity is None:
                        raise ValueError(
                            "Expected configured mock and nondecreasing logical time"
                        )
                    last_now = now
                else:
                    raise ValueError("Unsupported mock operation")
                response = {"op": op, "offline": True, "ros_domain_id": None}
                if op == "status":
                    reported = dict(identity)
                    if behavior == "identity_mismatch":
                        reported["configuration_id"] += ":changed"
                    response.update(
                        interface=reported,
                        consistent=behavior != "inconsistent",
                        owner_conflict=behavior == "ownership_conflict",
                        submitted_count=count,
                        last_wire=wire,
                    )
                elif op == "submit":
                    if behavior == "worker_crash":
                        os._exit(2)
                    if behavior == "ipc_failure":
                        channel.send(b'{"version":999}')
                        return
                    if behavior == "hang":
                        time.sleep(30)
                    action = request["action"]
                    values = action["values"]
                    arm = identity["kind"] == "fr3_normalized_pose_offset_v1"
                    if (
                        action["interface"] != identity
                        or action["kind"] != identity["kind"]
                        or action["session_id"] != session
                        or action["authority_id"] != authority
                        or type(values) is not list
                        or len(values) != (6 if arm else 1)
                        or not all(
                            type(v) in (int, float)
                            and math.isfinite(v)
                            and (-1 if arm else 0) <= v <= 1
                            for v in values
                        )
                        or not action["generated_ns"] <= now < action["deadline_ns"]
                    ):
                        raise ValueError(
                            "Expected a valid, unexpired request for the configured mock"
                        )
                    if type(request.get("admission_deadline_ns")) is not int:
                        raise ValueError("Expected derived local admission deadline")
                    if now >= request["admission_deadline_ns"]:
                        response.update(
                            state="rejected", reason="mock_admission_expired"
                        )
                    elif behavior in (
                        "reject",
                        "ownership_conflict",
                        "inconsistent",
                        "identity_mismatch",
                    ):
                        response.update(
                            state="rejected", reason="mock_submission_rejected"
                        )
                    else:
                        if active:
                            response["preempted_sequence"] = active["sequence"]
                        count += 1
                        wire = (
                            {
                                "linear": dict(zip(("x", "y", "z"), values[:3])),
                                "angular": dict(zip(("x", "y", "z"), values[3:])),
                            }
                            if arm
                            else {"command": {"position": values[0], "max_effort": 0.0}}
                        )
                        active = {
                            "sequence": action["sequence"],
                            "deadline_ns": action["deadline_ns"],
                            "due_ns": now + delay_ns,
                            "arm": arm,
                            "state": "acceptance_unknown"
                            if arm or behavior == "drop"
                            else "mock_acknowledged",
                            "closure": None if arm else values[0],
                        }
                        response.update(
                            state=active["state"],
                            wire=wire,
                            sequence=active["sequence"],
                        )
                elif op == "poll":
                    if active:
                        state = active["state"]
                        if now >= active["deadline_ns"]:
                            state = "expired"
                        elif behavior == "preempt":
                            state = "mock_canceled"
                        elif behavior == "fail":
                            state = "failed"
                        elif (
                            not active["arm"]
                            and behavior != "drop"
                            and now >= active["due_ns"]
                        ):
                            state = "mock_completed"
                        response.update(state=state, sequence=active["sequence"])
                        if state == "mock_completed":
                            response["result"] = {
                                "position": active["closure"],
                                "effort": None,
                                "reached_goal": True,
                                "stalled": False,
                                "physical_grasp_success": None,
                            }
                        if state in (
                            "expired",
                            "mock_canceled",
                            "failed",
                            "mock_completed",
                        ):
                            active = None
                elif op == "cancel":
                    if active and active["sequence"] == request["sequence"]:
                        active = None
                    response["state"] = "mock_canceled"
                send_packet(channel, response)
        except (ConnectionError, BrokenPipeError):
            return


if __name__ == "__main__":
    main()
