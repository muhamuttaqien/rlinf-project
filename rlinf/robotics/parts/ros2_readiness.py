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

"""Passive evidence assessment; this module cannot authorize physical commands."""

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Real
from typing import Any

from .ros2_observation import FR3ROS2Observation, RobotiqROS2Observation
from .transports.ros2.passive import COMMAND_TOPIC, GRIPPER_ACTIONS, PARAMETERS
from .transports.ros2.protocol import FR3_JOINTS, PROFILES

EXPECTED = {
    "fr3": {
        "domain": 70,
        "controller": "joint_impedance_ik_controller",
        "type": "franka_arm_controllers/JointImpedanceIKController",
        "hardware": "FrankaHardwareInterface",
        "claims": [f"fr3_joint{i}/effort" for i in range(1, 8)],
        "parameters": {
            "/controller_manager": {"update_rate": 1000, "use_sim_time": False},
            "/joint_impedance_ik_controller": {
                "ik_update_rate": 50.0,
                "command_timeout": 0.25,
                "arm_mounting_orientation": [0.0, 0.0, 0.0],
                "arm_id": "fr3",
                "load_gripper": "false",
                "use_sim_time": False,
                "k_gains": [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0],
                "d_gains": [30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0],
            },
        },
    },
    "robotiq": {
        "domain": 71,
        "controller": "robotiq_gripper_controller",
        "type": "robotiq_controllers/GripperCommandController",
        "hardware": "RobotiqGripperHardwareInterface",
        "claims": ["robotiq_85_left_knuckle_joint/position"],
        "parameters": {
            "/controller_manager": {"update_rate": 500, "use_sim_time": False},
            "/robotiq_gripper_controller": {
                "gripper_profile": "2f_85",
                "gripper_closed_position": 0.7929,
                "max_effort": 50.0,
                "allow_stalling": True,
                "use_object_status": True,
                "object_status_timeout": 10.0,
                "goal_tolerance": 0.02,
                "stall_timeout": 0.05,
                "action_monitor_rate": 100.0,
                "use_sim_time": False,
            },
        },
    },
}
CATEGORIES = (
    "observation_readiness",
    "controller_status_visibility",
    "gripper_status_visibility",
    "interface_identity_consistency",
    "timing_evidence_availability",
    "ownership_evidence_availability",
    "physical_safety_qualification",
)


def check(state: str, detail: str) -> dict[str, str]:
    """Record one passive finding without granting command authority."""
    return {"state": state, "detail": detail}


def summarize(checks: list[dict[str, str]]) -> dict[str, Any]:
    """Retain all findings and prefer failure or missing evidence over pass."""
    state = next(
        (
            s
            for s in ("FAIL", "UNKNOWN", "NOT_QUALIFIED")
            if any(c["state"] == s for c in checks)
        ),
        "PASS",
    )
    return {"state": state, "checks": checks}


def valid_sample(data: dict[str, Any], profile: str, name: str) -> bool:
    """Check decoded observation dimensions without inventing absent fields."""
    fields = {"position": 1}
    if profile == "fr3":
        fields = (
            {"position": 7, "velocity": 7, "effort": 7}
            if name == "joints"
            else {"tcp_pose": 7}
            if name == "pose"
            else {"force": 3, "torque": 3}
        )
        if name == "joints" and data.get("names") != list(FR3_JOINTS):
            raise ValueError(
                "Expected explicitly ordered fr3_joint1 through fr3_joint7"
            )
    for field, size in fields.items():
        values = data.get(field)
        if values is None:
            return False
        if len(values) != size or any(
            isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
            for x in values
        ):
            raise ValueError(f"Expected {size} finite numeric values in {field}")
    if (
        profile == "fr3"
        and name == "pose"
        and abs(sum(x * x for x in data["tcp_pose"][3:]) - 1) > 1e-3
    ):
        raise ValueError("Expected unit quaternion in x,y,z,w order")
    if (
        profile == "robotiq"
        and name == "normalized"
        and not 0 <= data["position"][0] <= 1
    ):
        raise ValueError("Expected normalized closure in [0,1]")
    return True


def assess_readiness(
    observations: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    max_age: float = 0.25,
    now_ns: int | None = None,
    wall_ns: int | None = None,
) -> dict[str, Any]:
    """Assess same-host evidence, retaining unknowns and contradictory reports.

    Receipt times must come from this host's monotonic clock. Source ROS times
    use a different clock; their wall-clock differences are evidence only.
    PASS describes the named passive category, never physical safety.
    """
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("Expected positive finite observation max_age")
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    wall_ns = time.time_ns() if wall_ns is None else wall_ns
    categories: dict[str, list] = {name: [] for name in CATEGORIES}
    timing, fingerprints = {}, {}
    for profile, expected in EXPECTED.items():

        def add(category: str, state: str, detail: str) -> None:
            categories[category].append(check(state, f"{profile}: {detail}"))

        try:
            observation = observations.get(profile, {})
            status = observation.get("ros2", {})
            envelope = diagnostics.get(profile, {})
            evidence = envelope.get("evidence", {})
            identity = "interface_identity_consistency"
            obs_checks = []
            stamps = []
            timing[profile] = {}
            if status.get("domain_id") != expected["domain"]:
                add(identity, "FAIL", "observation domain missing or inconsistent")
            if (envelope.get("domain_id"), envelope.get("profile")) != (
                expected["domain"],
                profile,
            ):
                add(
                    identity,
                    "UNKNOWN" if not envelope else "FAIL",
                    "diagnostic identity missing or inconsistent",
                )
            for name, (topic, kind, _) in PROFILES[profile].items():
                stream = status.get("streams", {}).get(name, {})
                sample = stream.get("sample")
                valid = (
                    bool(sample)
                    and stream.get("connected") is True
                    and not (stream.get("error") or stream.get("graph_error"))
                )
                if stream.get("topic") != topic or stream.get("type") != kind:
                    add(identity, "FAIL", f"{name} observation endpoint/type mismatch")
                if sample:
                    valid = valid and valid_sample(sample["data"], profile, name)
                    received = sample["received_monotonic_ns"]
                    if type(received) is not int:
                        raise ValueError("Expected integer local receipt timestamp")
                    age = (now_ns - received) / 1e9
                    header = sample["header"]
                    sec, nanos = header["sec"], header["nanosec"]
                    if (
                        type(sec) is not int
                        or type(nanos) is not int
                        or sec < 0
                        or not 0 <= nanos < 10**9
                    ):
                        raise ValueError("Expected valid ROS header timestamp")
                    stamp = sec * 10**9 + nanos
                    timing[profile][name] = {
                        "receipt_age_s": age,
                        "source_stamp_ns": stamp,
                        "source_wall_difference_s_unqualified": (wall_ns - stamp) / 1e9,
                        "frame_id": header["frame_id"],
                        "sequence": stream["sequence"],
                    }
                    if stamp > 0:
                        stamps.append(stamp)
                    valid = valid and 0 <= age <= max_age
                obs_checks.append(bool(valid))
            complete = status.get("complete") is True and all(obs_checks)
            add(
                "observation_readiness",
                "PASS" if complete else "FAIL",
                "selected streams complete, valid and fresh"
                if complete
                else "missing/stale/invalid/disconnected observations",
            )
            add(
                "timing_evidence_availability",
                "PASS" if complete and len(stamps) == len(obs_checks) else "UNKNOWN",
                "source headers and local receipt ages; no qualified delay or clock-error bound",
            )
            timing[profile]["source_stamp_spread_s"] = (
                (max(stamps) - min(stamps)) / 1e9 if stamps else None
            )
            services = evidence.get("services", {})
            controllers = services.get("/controller_manager/list_controllers")
            hardware = services.get("/controller_manager/list_hardware_components")
            interfaces = services.get("/controller_manager/list_hardware_interfaces")
            if controllers is None or hardware is None or interfaces is None:
                add(
                    "controller_status_visibility",
                    "UNKNOWN",
                    "required listing service evidence missing",
                )
                add(
                    identity,
                    "UNKNOWN",
                    "controller/resource identity cannot be fully cross-checked",
                )
            else:
                selected = [
                    x for x in controllers if x["name"] == expected["controller"]
                ]
                components = [x for x in hardware if x["name"] == expected["hardware"]]
                if len(selected) != 1 or len(components) != 1:
                    add(
                        "controller_status_visibility",
                        "FAIL",
                        "expected controller/hardware missing or duplicated",
                    )
                    add(
                        identity,
                        "FAIL",
                        "expected controller/hardware identity missing or duplicated",
                    )
                else:
                    controller, component = selected[0], components[0]
                    consistent = (
                        controller["state"] == "active"
                        and component["state"] == "active"
                        and component["state_id"] == 3
                    )
                    consistent &= sorted(controller["claimed"]) == sorted(
                        expected["claims"]
                    )
                    consistent &= all(
                        sum(claim in x["claimed"] for x in controllers) == 1
                        for claim in expected["claims"]
                    )
                    for source in (interfaces, component["command"]):
                        for claim in expected["claims"]:
                            found = [x for x in source if x["name"] == claim]
                            consistent &= (
                                len(found) == 1
                                and found[0]["claimed"] is True
                                and found[0]["available"] is True
                            )
                    add(
                        "controller_status_visibility",
                        "PASS" if consistent else "FAIL",
                        "active state and expected claims agree"
                        if consistent
                        else "controller/component/interface state or claims conflict",
                    )
                    add(
                        identity,
                        "PASS" if controller["type"] == expected["type"] else "FAIL",
                        "reported controller plugin type",
                    )
            for target, parameters in expected["parameters"].items():
                actual = services.get(target + "/get_parameters", {})
                for key, value in parameters.items():
                    state = (
                        "UNKNOWN"
                        if actual.get(key) is None
                        else "PASS"
                        if actual[key] == value and type(actual[key]) is type(value)
                        else "FAIL"
                    )
                    # ROS integer/double representations remain explicit in the fingerprint.
                    add(identity, state, f"configuration {target}:{key}")
            subset = {
                target: services.get(target + "/get_parameters")
                for target in PARAMETERS[profile]
            }
            fingerprints[profile] = hashlib.sha256(
                json.dumps(subset, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            topics = evidence.get("topics", {})
            for topic, kind, _ in PROFILES[profile].values():
                publishers = topics.get(topic, {}).get("publishers", [])
                add(
                    identity,
                    "UNKNOWN"
                    if not publishers
                    else "PASS"
                    if all(x["type"] == kind for x in publishers)
                    else "FAIL",
                    f"state publisher type {topic}",
                )
                qos_known = publishers and all(
                    x["reliability"] in ("RELIABLE", "BEST_EFFORT")
                    and x["durability"] in ("VOLATILE", "TRANSIENT_LOCAL")
                    for x in publishers
                )
                add(
                    "observation_readiness",
                    "PASS" if qos_known else "UNKNOWN",
                    f"publisher QoS visible/compatible with best-effort volatile reader: {topic}",
                )
            nodes = evidence.get("nodes", [])
            managers = [
                x
                for x in nodes
                if x == ["controller_manager", "/"] or x == ("controller_manager", "/")
            ]
            add(
                "controller_status_visibility",
                "FAIL"
                if len(managers) > 1
                else "PASS"
                if len(managers) == 1
                else "UNKNOWN",
                "controller-manager graph identity count",
            )
            if profile == "fr3":
                endpoint = topics.get(COMMAND_TOPIC, {})
                subscribers = endpoint.get("subscriptions", [])
                matching = [
                    x
                    for x in subscribers
                    if x["node"] == expected["controller"]
                    and x["type"] == "geometry_msgs/msg/Twist"
                ]
                unknown_receiver = (
                    len(subscribers) == 1
                    and subscribers[0]["node"] == "_NODE_NAME_UNKNOWN_"
                    and subscribers[0]["type"] == "geometry_msgs/msg/Twist"
                )
                add(
                    identity,
                    "PASS"
                    if len(subscribers) == len(matching) == 1
                    else "FAIL"
                    if subscribers and not unknown_receiver
                    else "UNKNOWN",
                    "Twist endpoint and receiving controller",
                )
                writers = endpoint.get("publishers", [])
                add(
                    "ownership_evidence_availability",
                    "FAIL"
                    if len(writers) > 1
                    else "UNKNOWN"
                    if writers or len(matching) != 1
                    else "PASS",
                    f"{len(writers)} discovered arm writers; graph absence grants no lease",
                )
            else:
                servers = evidence.get("action_servers", [])
                servers_visible = True
                for endpoint in GRIPPER_ACTIONS:
                    found = [x for x in servers if x["endpoint"] == endpoint]
                    good = (
                        len(found) == 1
                        and found[0]["types"] == ["control_msgs/action/GripperCommand"]
                        and found[0]["node"] == expected["controller"]
                    )
                    servers_visible &= good
                    add(
                        identity,
                        "PASS" if good else "FAIL" if found else "UNKNOWN",
                        f"action server {endpoint}",
                    )
                clients = evidence.get("action_clients")
                owners = {
                    (x["node"], x["namespace"])
                    for x in clients or []
                    if (x["node"], x["namespace"]) != (expected["controller"], "/")
                }
                add(
                    "ownership_evidence_availability",
                    "UNKNOWN"
                    if clients is None or not servers_visible
                    else "FAIL"
                    if len(owners) > 1
                    else "PASS",
                    f"{len(owners)} external gripper client identities; controller's internal relay is recorded separately; handover remains unqualified",
                )
            health = evidence.get("health", {})
            sample = health.get("sample")
            category = (
                "gripper_status_visibility"
                if profile == "robotiq"
                else "controller_status_visibility"
            )
            if not sample or health.get("error"):
                add(
                    category,
                    "UNKNOWN",
                    "typed health unavailable: " + str(health.get("error")),
                )
            elif not 0 <= (now_ns - sample["received_monotonic_ns"]) / 1e9 <= max_age:
                add(
                    category,
                    "FAIL",
                    "typed health receipt is stale or from an invalid clock",
                )
            else:
                data = sample["data"]
                if profile == "robotiq":
                    unknown = 255 in (
                        data["gripper_fault"],
                        data["gripper_fault_severity"],
                    )
                    healthy = (
                        data["gripper_fault"] == data["gripper_fault_severity"] == 0
                    )
                    add(
                        category,
                        "UNKNOWN" if unknown else "PASS" if healthy else "FAIL",
                        "source-reported gripper fault/severity; current is not force",
                    )
                else:
                    healthy = (
                        data["robot_mode"] in (1, 2, 3)
                        and data["current_errors"]
                        and not any(data["current_errors"].values())
                    )
                    add(
                        category,
                        "PASS" if healthy else "FAIL",
                        "source-reported mode/current errors; not a physical safety verdict",
                    )
            if evidence.get("errors"):
                add(
                    "controller_status_visibility",
                    "UNKNOWN",
                    "diagnostic errors: " + str(evidence["errors"])[:2048],
                )
                if "collection" in evidence["errors"]:
                    add(identity, "UNKNOWN", "graph/status collection incomplete")
                    add(
                        "ownership_evidence_availability",
                        "UNKNOWN",
                        "graph/status collection incomplete",
                    )
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
            for category in CATEGORIES[:-1]:
                add(category, "FAIL", f"malformed evidence: {exc}")
    categories["physical_safety_qualification"].append(
        check(
            "NOT_QUALIFIED", "Operator-approved physical qualification is outstanding"
        )
    )
    source_stamps = [
        stream["source_stamp_ns"]
        for streams in timing.values()
        for stream in streams.values()
        if isinstance(stream, dict) and stream.get("source_stamp_ns", 0) > 0
    ]
    return {
        "schema": "fr3_passive_readiness_v1",
        "assessment_monotonic_ns": now_ns,
        "assessment_wall_ns": wall_ns,
        "monotonic_clock_domain": "preflight_host",
        "physical_command_authorization": "NOT_GRANTED",
        "qualification_gate": "NO_GO",
        "categories": {name: summarize(values) for name, values in categories.items()},
        "timing": timing,
        "cross_stream_source_stamp_spread_s_unqualified": (
            max(source_stamps) - min(source_stamps)
        )
        / 1e9
        if source_stamps
        else None,
        "configuration_subset_sha256": fingerprints,
        "binary_identity": "UNKNOWN: requires independent process/library attestation",
        "timing_qualification": "UNKNOWN: no host-to-host or worst-case latency bound",
        "exclusive_authority": "NOT_GRANTED: graph discovery is not an ownership lease",
        "observations": observations,
        "diagnostics": diagnostics,
    }


def run_preflight(
    *,
    duration: float = 2.0,
    budget: float = 2.0,
    max_age: float = 0.25,
    timeout: float = 6.0,
    ros_python: str = "/usr/bin/python3",
    ros_setup: str = "/opt/ros/jazzy/setup.bash",
) -> dict[str, Any]:
    """Own two bounded passive workers, then release only those resources.

    Duration allows graph/state discovery before a shared-budget query per
    domain. Timeouts bound each startup/IPC request; partial failures are data.
    """
    if not all(math.isfinite(v) for v in (duration, budget, max_age, timeout)) or not (
        0 < duration <= 30
        and 0 < budget <= 5
        and budget + 0.5 <= timeout <= 30
        and max_age > 0
    ):
        raise ValueError(
            "Expected duration (0,30], budget (0,5], timeout [budget+0.5,30], positive max_age"
        )
    parts = {
        name: cls(
            max_age=max_age, timeout=timeout, ros_python=ros_python, ros_setup=ros_setup
        )
        for name, cls in (
            ("fr3", FR3ROS2Observation),
            ("robotiq", RobotiqROS2Observation),
        )
    }
    observations, diagnostics, errors = {}, {}, {}
    started = time.monotonic_ns()
    try:
        for name, part in parts.items():
            try:
                part.connect()
            except Exception as exc:
                errors[name] = str(exc)[:2048]
        time.sleep(duration)

        def inspect(part: Any) -> dict[str, Any]:
            before = time.monotonic_ns()
            result = part.inspect_status(budget)
            result["ipc_round_trip_s"] = (time.monotonic_ns() - before) / 1e9
            return result

        # Concurrent domain queries keep the earlier health snapshot from aging
        # through another domain's entire response window.
        with ThreadPoolExecutor(max_workers=2) as pool:
            queries = {
                pool.submit(inspect, part): name
                for name, part in parts.items()
                if part.is_connected
            }
            for future in as_completed(queries):
                name = queries[future]
                try:
                    diagnostics[name] = future.result()
                except Exception as exc:
                    errors[name] = str(exc)[:2048]
        for name, part in parts.items():
            observations[name] = part.get_state()
        report = assess_readiness(observations, diagnostics, max_age=max_age)
        report["errors"] = errors
    finally:
        for part in parts.values():
            part.disconnect()
    report["elapsed_s"] = (time.monotonic_ns() - started) / 1e9
    return report
