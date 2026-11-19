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

"""Fixed NUC-side passive program, bundled over SSH without remote installation."""

import hashlib
import os
import time
from pathlib import Path
from typing import Any

from passive import collect_status, origin

SCHEMA = "fr3_nuc_passive_v1"
SNAPSHOTS = 3
BUDGET = 2.0
LIBRARIES = {
    "libcontroller_manager.so",
    "libhardware_interface.so",
    "libfranka_hardware.so",
    "libfranka_arm_controllers.so",
    "librobotiq_controllers.so",
    "librobotiq_driver.so",
}


def file_hash(path: Path) -> str:
    """Hash an identified library with a finite size limit."""
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Expected mapped diagnostic library <=32 MiB")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manager_processes(
    domain: int, proc_root: Path = Path("/proc")
) -> list[dict[str, Any]]:
    """Read only manager identity fields and whitelisted mapped-library metadata."""
    result = []
    for process in proc_root.glob("[0-9]*"):
        try:
            if process.stat().st_uid != os.getuid():
                continue
            executable = (process / "exe").resolve(strict=True)
            if executable.name != "ros2_control_node":
                continue
            environment = (process / "environ").read_bytes().split(b"\0")
            if f"ROS_DOMAIN_ID={domain}".encode() not in environment:
                continue
            argv = (process / "cmdline").read_bytes()
            stat = (process / "stat").read_text().rsplit(")", 1)[1].split()
            mappings = (process / "maps").read_text()
            libraries = {}
            for line in mappings.splitlines():
                columns = line.split(None, 5)
                if len(columns) != 6:
                    continue
                path = Path(columns[5])
                if path.name in LIBRARIES and str(path) not in libraries:
                    libraries[str(path)] = {
                        "sha256": file_hash(path),
                        "mapped_inode": int(columns[4]),
                        "device": columns[3],
                        "file_inode": path.stat().st_ino,
                    }
            result.append(
                {
                    "pid": int(process.name),
                    "ppid": int(stat[1]),
                    "start_ticks": int(stat[19]),
                    "uid": process.stat().st_uid,
                    "executable": str(executable),
                    "domain_id": domain,
                    "argv_sha256": hashlib.sha256(argv).hexdigest(),
                    "libraries": libraries,
                }
            )
            if len(result) > 8:
                raise ValueError("Expected at most 8 manager processes per domain")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return sorted(result, key=lambda x: x["pid"])


def collect(nonce: str) -> dict[str, Any]:
    """Take exactly three snapshots per fixed domain; never accept ROS operations."""
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "nonce": nonce,
        "origin": origin(),
        "started_wall_ns": time.time_ns(),
        "snapshots": [],
        "physical_command_authorization": "NOT_GRANTED",
        "qualification_gate": "NO_GO",
        "process_inventory_scope": "readable same-uid processes; graph correlation is not cryptographic PID attribution",
    }
    if os.environ.get("ROS_DISTRO") != "humble":
        raise ValueError("Expected existing NUC ROS 2 Humble runtime")
    for domain, profile in ((70, "fr3"), (71, "robotiq")):
        context = Context()
        node = executor = None
        try:
            os.environ["ROS_DOMAIN_ID"] = str(domain)
            rclpy.init(args=[], context=context, domain_id=domain)
            node = Node(
                "rlinf_2d_passive_probe",
                context=context,
                enable_rosout=False,
                start_parameter_services=False,
                use_global_arguments=False,
            )
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            for index in range(SNAPSHOTS):
                before = manager_processes(domain)
                evidence = collect_status(
                    node, executor, profile, BUDGET, include_observations=True
                )
                after = manager_processes(domain)
                report["snapshots"].append(
                    {
                        "index": index,
                        "profile": profile,
                        "domain_id": domain,
                        "manager_before": before,
                        "manager_after": after,
                        "evidence": evidence,
                    }
                )
                if index + 1 < SNAPSHOTS:
                    time.sleep(0.1)
        finally:
            if executor is not None:
                executor.shutdown(timeout_sec=1.0)
            if node is not None:
                node.destroy_node()
            context.try_shutdown()
    report["completed_wall_ns"] = time.time_ns()
    return report
