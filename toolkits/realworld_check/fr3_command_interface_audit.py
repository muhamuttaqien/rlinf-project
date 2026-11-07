# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""FR3/Robotiq ROS2 command-interface DISCOVERY ONLY.

Queries topic endpoints, a controller parameter and gripper action discovery.
No publisher, goal client, service mutator, controller command, safety approval
or enable mechanism exists here. Results never certify autonomous actuation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from typing import Any

ARM_TOPIC = "/franka_controller/target_cartesian_velocity_percent"
GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"
ARM_CONTROLLER = "/joint_impedance_ik_controller"
EXPECTED_TIMEOUT_S = 0.25


def parse_topic_info(output: str) -> dict[str, Any]:
    def number(name: str) -> int | None:
        match = re.search(rf"(?im)^\s*{name} count:\s*(\d+)\s*$", output)
        return int(match.group(1)) if match else None

    match = re.search(r"(?im)^\s*Type:\s*(\S+)\s*$", output)
    nodes = re.findall(r"(?im)^\s*Node name:\s*(\S+)\s*$", output)
    return {
        "message_type": match.group(1) if match else None,
        "publishers": number("Publisher"),
        "subscriptions": number("Subscription"),
        "reported_nodes": sorted(set(nodes)),
    }


def parse_controller_timeout(output: str) -> float | None:
    match = re.search(
        r"(?i)(?:Double value is:|Value:)\s*([0-9]+(?:\.[0-9]+)?)",
        output,
    )
    return float(match.group(1)) if match else None


def parse_actions(output: str) -> dict[str, str]:
    found = {}
    for line in output.splitlines():
        match = re.fullmatch(r"\s*(/\S+)\s+\[([a-zA-Z0-9_/]+)\]\s*", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def discovery_report(
    topic_text: str, timeout_text: str, actions_text: str,
    *, command_errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    topic = parse_topic_info(topic_text)
    timeout = parse_controller_timeout(timeout_text)
    actions = parse_actions(actions_text)
    expected_action_type = "control_msgs/action/GripperCommand"
    errors = command_errors or {}
    return {
        "mode": "read_only_interface_discovery",
        "arm_domain_id": 70,
        "robotiq_domain_id": 71,
        "arm_topic": ARM_TOPIC,
        "arm_topic_info": topic,
        "arm_topic_type_matches": topic["message_type"] == "geometry_msgs/msg/Twist",
        "arm_controller_endpoint_reported": any(
            "joint_impedance_ik_controller" in name for name in topic["reported_nodes"]
        ),
        "arm_motion_publishers_count": topic["publishers"],
        "exclusive_publish_source_verified": (
            False  # Node presence/count is not proof of control ownership.
        ),
        "command_timeout_s": timeout,
        "command_timeout_matches_last_known": (
            timeout is not None and abs(timeout - EXPECTED_TIMEOUT_S) < 1e-9
        ),
        "timeout_means": "hold last impedance reference, NOT an emergency stop",
        "controller_motion_input_semantics": (
            "6D normalized SpaceMouse Twist [-1,1], IK relative pose update; "
            "not SI velocities or the proposed 7D policy delta"
        ),
        "reference_cartesian_offset_scales": {
            "linear_m_per_normalized_axis": 0.007,
            "angular_rad_per_normalized_axis": 0.03,
            "source": "previously inspected NUC C++ implementation; not proven by ROS graph",
        },
        "gripper_action": GRIPPER_ACTION,
        "gripper_action_type_observed": actions.get(GRIPPER_ACTION),
        "gripper_action_type_matches": actions.get(GRIPPER_ACTION) == expected_action_type,
        "gripper_command_semantics": (
            "previously observed bridge: binary open/close -> knuckle 0.0/0.7929 "
            "(unit/joint convention and goal completion not yet qualified)"
        ),
        "command_query_errors": errors,
        "graph_discovery_complete": (
            not errors and topic["message_type"] == "geometry_msgs/msg/Twist"
            and topic["publishers"] is not None
            and topic["subscriptions"] is not None
            and topic["subscriptions"] >= 1
            and timeout is not None
            and actions.get(GRIPPER_ACTION) == expected_action_type
        ),
        "policy_action_mapping_verified": False,
        "independent_controller_safety_verified": False,
        "policy_command_bridge_present": False,
        "robot_commands_sent": False,
        "physical_execution_approved": False,
        "next_step": (
            "Review current NUC controller/Robotiq source and approve a separate "
            "policy command API with exclusive control, units/frames, bounds, "
            "watchdog and safety sign-off before building physical actuation"
        ),
    }


def _read_only_query(
    args: list[str], domain: int, timeout_s: int, ssh_host: str | None = None,
) -> str:
    # The three permitted args lists are fixed by main(), not user input.
    # SSH is used to query the ROS Humble installation on the NUC instead
    # of the Jazzy CLI whose metadata import failed in the workstation env.
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain)
    command = args
    options: dict[str, Any] = {"env": env}
    if ssh_host is not None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", ssh_host):
            raise ValueError("SSH host must be a simple configured host alias")
        remote_script = "\n".join([
            "set -e",
            "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV",
            "source /opt/ros/humble/setup.bash",
            'if [ -f "$HOME/Desktop/aist-gamma/franka_controller_ws/install/setup.bash" ]; then',
            '  source "$HOME/Desktop/aist-gamma/franka_controller_ws/install/setup.bash"',
            "fi",
            f"export ROS_DOMAIN_ID={domain}",
            "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
            shlex.join(args),
            "",
        ])
        command = [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            ssh_host, "bash", "-s",
        ]
        options["input"] = remote_script
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            timeout=timeout_s, check=False, **options,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "SSH or ros2 command not found; check local PATH/SSH alias"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ROS domain {domain} query timed out"
            + (" via SSH" if ssh_host else "")
        ) from exc
    if result.returncode != 0:
        # Preserve the useful END of traceback (PackageNotFoundError etc)
        # instead of the original first 250 characters.
        error_tail = (result.stderr or result.stdout).strip()[-1800:]
        raise RuntimeError(
            f"ROS domain {domain} query failed ({result.returncode})"
            f"{' via SSH' if ssh_host else ''}: {error_tail}"
        )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--ssh-host", default=None,
                        help="Run read-only ROS Humble discovery on NUC via SSH alias")
    args = parser.parse_args()
    if args.timeout_seconds < 1 or args.timeout_seconds > 60:
        parser.error("timeout-seconds must be between 1 and 60")

    queries = {
        "arm_topic": (["ros2", "topic", "info", "-v", ARM_TOPIC], 70),
        "arm_timeout": (
            ["ros2", "param", "get", ARM_CONTROLLER, "command_timeout"], 70
        ),
        "gripper_actions": (["ros2", "action", "list", "-t"], 71),
    }
    output = {}
    errors = {}
    for name, (argv, domain) in queries.items():
        try:
            output[name] = _read_only_query(
                argv, domain, args.timeout_seconds, ssh_host=args.ssh_host
            )
        except RuntimeError as exc:
            output[name] = ""
            errors[name] = str(exc)
    report = discovery_report(
        output["arm_topic"], output["arm_timeout"], output["gripper_actions"],
        command_errors=errors,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["graph_discovery_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
