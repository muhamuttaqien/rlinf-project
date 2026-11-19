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

"""4C.2T: offline validation of *proposed* FR3 controller telemetry.

Pure Python; no ROS imports, control messages, action labels, LeRobot export,
physical FK assumptions, or hardware access. Existing raw captures do not
contain these records, and must never be retrofitted with invented targets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "fr3_controller_target_telemetry_proposal_v1"
GATES = (
    "moving", "intent_fresh", "target_valid", "activation_matches",
    "state_fresh", "command_fresh",
)
ARM_KEYS = {
    "schema", "event", "clock_id", "seq", "tick_ns", "activation_ns",
    "intent_recv_ns", "target_state_ns", "target_command_ns",
    "target_activation_ns", "sample_ns", "ik_solve_ns", "intent_axes",
    "ik_target_q_rad", "reference_before_rad", "reference_after_rad",
    "measured_q_rad", "velocity_limits_rad_s", "dt_s",
    "command_timeout_s", "gates", "reference_advanced",
    "telemetry_drops_total",
}
GRIPPER_KEYS = {
    "schema", "event", "clock_id", "seq", "goal_id", "request_ns",
    "sent_ns", "accepted_ns", "result_ns", "normalized_request",
    "goal_target_knuckle_rad", "max_effort", "accepted", "result_status",
    "measured_knuckle_rad", "measured_ns", "replaced_requests_since_last_goal",
}


def _keys(record: Mapping[str, Any], expected: set[str]) -> None:
    if set(record) != expected:
        raise ValueError(f"Missing/unexpected fields: {sorted(expected ^ set(record))}")


def _integer(value: object, label: str, *, zero_ok: bool = False) -> int:
    if type(value) is not int or value < (0 if zero_ok else 1):
        raise ValueError(f"{label} must be a {'nonnegative' if zero_ok else 'positive'} integer")
    return value


def _scalar(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _vector(record: Mapping[str, Any], field: str, n: int) -> list[float]:
    value = record[field]
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{field} must contain {n} numeric values")
    return [_scalar(x, field) for x in value]


def _optional_time(value: object, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _check_arm(r: Mapping[str, Any]) -> None:
    _keys(r, ARM_KEYS)
    tick = _integer(r["tick_ns"], "tick_ns")
    activation = _integer(r["activation_ns"], "activation_ns")
    stamps = ("intent_recv_ns", "target_state_ns", "target_command_ns",
              "sample_ns", "ik_solve_ns")
    for key in stamps:
        stamp = _integer(r[key], key)
        if stamp > tick:
            raise ValueError(f"{key} is in the future relative to tick_ns")
    if activation > tick:
        raise ValueError("activation_ns is in the future")
    target_activation = _integer(r["target_activation_ns"], "target_activation_ns")
    axes = _vector(r, "intent_axes", 6)
    if any(abs(x) > 1.0 for x in axes):
        raise ValueError("intent_axes must be normalized to [-1, 1]")
    before = _vector(r, "reference_before_rad", 7)
    after = _vector(r, "reference_after_rad", 7)
    _vector(r, "measured_q_rad", 7)
    limits = _vector(r, "velocity_limits_rad_s", 7)
    if any(x <= 0.0 for x in limits):
        raise ValueError("velocity_limits_rad_s must be positive")
    dt = _scalar(r["dt_s"], "dt_s")
    timeout = _scalar(r["command_timeout_s"], "command_timeout_s")
    if not 0.0 < dt <= 0.001 or timeout <= 0.0:
        raise ValueError("Invalid update dt or command timeout")
    _integer(r["telemetry_drops_total"], "telemetry_drops_total", zero_ok=True)
    gates = r["gates"]
    if not isinstance(gates, dict) or set(gates) != set(GATES):
        raise ValueError("Missing or unknown freshness/validity gates")
    if any(type(v) is not bool for v in gates.values()):
        raise ValueError("All gates must be booleans")
    if gates["moving"] != any(x != 0.0 for x in axes):
        raise ValueError("moving gate contradicts normalized intent")
    if gates["activation_matches"] != (target_activation == activation):
        raise ValueError("Activation-generation gate mismatch")
    def fresh(stamp: int) -> bool:
        return stamp >= activation and (tick - stamp) * 1e-9 <= timeout
    for gate, field in (("intent_fresh", "intent_recv_ns"),
                        ("state_fresh", "target_state_ns"),
                        ("command_fresh", "target_command_ns")):
        if gates[gate] != fresh(r[field]):
            raise ValueError(f"{gate} contradicts timestamp freshness")
    target = r["ik_target_q_rad"]
    if gates["target_valid"]:
        target = _vector(r, "ik_target_q_rad", 7)
    elif target is not None:
        raise ValueError("Invalid IK target must be recorded as null")
    advanced = r["reference_advanced"]
    if type(advanced) is not bool or advanced != all(gates.values()):
        raise ValueError("reference_advanced contradicts controller gates")
    for i in range(7):
        expected = before[i]
        if advanced:
            diff = target[i] - before[i]
            step = limits[i] * dt
            expected += max(-step, min(step, diff))
        if not math.isclose(after[i], expected, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"reference_after_rad[{i}] mismatches rate-limited update/hold")


def _check_gripper(r: Mapping[str, Any]) -> None:
    _keys(r, GRIPPER_KEYS)
    if not isinstance(r["goal_id"], str) or not r["goal_id"].strip():
        raise ValueError("goal_id must be a nonempty correlation identifier")
    request = _integer(r["request_ns"], "request_ns")
    sent = _integer(r["sent_ns"], "sent_ns")
    accepted_ns = _optional_time(r["accepted_ns"], "accepted_ns")
    result_ns = _optional_time(r["result_ns"], "result_ns")
    if sent < request or (accepted_ns is not None and accepted_ns < sent):
        raise ValueError("Gripper timestamps out of order")
    if result_ns is not None and (accepted_ns is None or result_ns < accepted_ns):
        raise ValueError("Gripper result timestamp out of order")
    _integer(r["replaced_requests_since_last_goal"], "replaced requests", zero_ok=True)
    request_fraction = _scalar(r["normalized_request"], "normalized_request")
    target = _scalar(r["goal_target_knuckle_rad"], "goal_target_knuckle_rad")
    if (request_fraction, target) not in ((1.0, 0.0), (0.0, 0.7929)):
        raise ValueError("Goal differs from observed binary Robotiq bridge mapping")
    if _scalar(r["max_effort"], "max_effort") != 50.0:
        raise ValueError("Unexpected bridge max_effort")
    accepted = r["accepted"]
    if accepted is not None and type(accepted) is not bool:
        raise ValueError("accepted must be bool or null")
    if (accepted is None) != (accepted_ns is None):
        raise ValueError("Acceptance result/timestamp mismatch")
    status = r["result_status"]
    if status is not None:
        _integer(status, "result_status", zero_ok=True)
    if (result_ns is None) != (status is None) or (result_ns is not None and accepted is not True):
        raise ValueError("Action result/acceptance mismatch")
    measured = r["measured_knuckle_rad"]
    measured_ns = _optional_time(r["measured_ns"], "measured_ns")
    if (measured is None) != (measured_ns is None):
        raise ValueError("Measured position requires measurement timestamp")
    if measured is not None:
        angle = _scalar(measured, "measured_knuckle_rad")
        if not 0.0 <= angle <= 0.8 or measured_ns < request:
            raise ValueError("Measured knuckle position/time invalid")


def audit_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate proposed offline telemetry; NEVER generate controller actions."""
    if not rows:
        raise ValueError("No telemetry records supplied")
    last_seq: dict[str, int] = {}
    last_tick: dict[str, int] = {}
    drops_last: dict[str, int] = {}
    goals_seen: set[str] = set()
    gaps = 0
    counts = {"arm_tick": 0, "gripper_goal": 0}
    advanced_count = 0
    for index, r in enumerate(rows):
        if not isinstance(r, dict) or r.get("schema") != SCHEMA:
            raise ValueError(f"Record {index}: unsupported telemetry schema")
        event = r.get("event")
        if event not in counts:
            raise ValueError(f"Record {index}: unsupported event type")
        if not isinstance(r.get("clock_id"), str) or not r["clock_id"].strip():
            raise ValueError(f"Record {index}: clock_id required")
        seq = _integer(r.get("seq"), "seq", zero_ok=True)
        key = r["clock_id"] + ":" + event
        if key in last_seq:
            if seq <= last_seq[key]:
                raise ValueError("Duplicate or out-of-order telemetry sequence")
            gaps += seq - last_seq[key] - 1
        last_seq[key] = seq
        if event == "arm_tick":
            _check_arm(r)
            if key in last_tick and r["tick_ns"] <= last_tick[key]:
                raise ValueError("Non-monotonic arm tick timestamp")
            if key in drops_last and r["telemetry_drops_total"] < drops_last[key]:
                raise ValueError("Cumulative telemetry drops decreased")
            drops_last[key] = r["telemetry_drops_total"]
            last_tick[key] = r["tick_ns"]
            advanced_count += int(r["reference_advanced"])
        else:
            _check_gripper(r)
            if r["goal_id"] in goals_seen:
                raise ValueError("Duplicate gripper goal_id")
            goals_seen.add(r["goal_id"])
        counts[event] += 1
    return {
        "schema": SCHEMA,
        "status": "offline_proposal_validated_not_approved",
        "record_count": len(rows),
        "arm_tick_count": counts["arm_tick"],
        "arm_reference_advanced_count": advanced_count,
        "arm_reference_held_count": counts["arm_tick"] - advanced_count,
        "gripper_goal_count": counts["gripper_goal"],
        "sequence_gaps": gaps,
        "last_arm_telemetry_drops_by_clock": {
            key.removesuffix(":arm_tick"): value for key, value in drops_last.items()
        },
        "cross_host_clock_alignment_verified": False,
        "tcp_fk_and_frame_semantics_verified": False,
        "arm_cartesian_action_labels_available": False,
        "7d_training_actions_available": False,
        "lerobot_export_ready": False,
        "sft_ready": False,
        "recap_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True,
                        help="Existing proposed-format JSONL (offline; never robot topics)")
    args = parser.parse_args()
    with args.jsonl.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    print(json.dumps(audit_telemetry(rows), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
