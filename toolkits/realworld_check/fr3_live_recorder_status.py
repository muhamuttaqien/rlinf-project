#!/usr/bin/env python3
"""NUC read-only FR3 controller/telemetry inspection. No ROS calls or writes.

A configuration file path alone does not establish that a controller plugin
was loaded, nor that its non-realtime JSONL recorder is active.
"""
from __future__ import annotations

import os
from pathlib import Path

PROC = Path("/proc")
ISOLATED = "fr3_telemetry_isolated_4c2t7"


def read_bytes(path):
    try:
        return path.read_bytes()
    except (OSError, PermissionError):
        return None


def parse_cmd(pid):
    raw = read_bytes(PROC / str(pid) / "cmdline")
    return [] if raw is None else [
        s.decode("utf-8", "replace") for s in raw.split(b"\0") if s
    ]


def parse_env(pid):
    raw = read_bytes(PROC / str(pid) / "environ")
    if raw is None:
        return None
    env = {}
    for entry in raw.split(b"\0"):
        if b"=" in entry:
            k, v = entry.split(b"=", 1)
            env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return env


def mapped_arm_plugins(pid):
    raw = read_bytes(PROC / str(pid) / "maps")
    if raw is None:
        return None
    found = set()
    for line in raw.decode("utf-8", "replace").splitlines():
        if "libfranka_arm_controllers.so" in line:
            columns = line.split(maxsplit=5)
            if len(columns) == 6:
                found.add(columns[5])
    return sorted(found)


def open_jsonl(pid):
    try:
        descriptors = list((PROC / str(pid) / "fd").iterdir())
    except (OSError, PermissionError):
        return None
    results = set()
    for fd in descriptors:
        try:
            name = os.readlink(fd)
        except (OSError, PermissionError):
            continue
        if ".jsonl" in name and not name.startswith("socket:"):
            results.add(name)
    return sorted(results)


def launch_logging_settings(argv):
    found = []
    for i in range(len(argv) - 1):
        if argv[i] != "--params-file":
            continue
        try:
            source = Path(argv[i + 1]).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        found += [line.strip()[:220] for line in source.splitlines()
                  if line.strip().startswith("telemetry_jsonl_path:")]
    return found


def inspect():
    processes = []
    for d in sorted(PROC.iterdir(), key=lambda x: int(x.name) if x.name.isdecimal() else -1):
        if not d.name.isdecimal():
            continue
        argv = parse_cmd(d.name)
        if not any("controller_manager/ros2_control_node" in x for x in argv):
            continue
        env = parse_env(d.name)
        maps = mapped_arm_plugins(d.name)
        fds = open_jsonl(d.name)
        processes.append({
            "pid": d.name,
            "domain": env.get("ROS_DOMAIN_ID", "<unset>") if env is not None else "<unreadable>",
            "config_ref": any(ISOLATED in x for x in argv),
            "libs": maps,
            "isolated_loaded": None if maps is None else any(ISOLATED in m for m in maps),
            "jsonl": fds,
            "setting": launch_logging_settings(argv),
        })
    return processes


def main():
    processes = inspect()
    if not processes:
        print("UNKNOWN: No readable running ROS controller-manager process")
        return 2
    for p in processes:
        print(f"\nPID={p['pid']} DOMAIN={p['domain']}")
        print(f"  Isolated config referenced: {p['config_ref']}")
        print(f"  Loaded arm controller library: {p['libs'] if p['libs'] is not None else 'UNREADABLE'}")
        print(f"  Isolated controller library loaded: {p['isolated_loaded']}")
        print(f"  JSONL file open: {p['jsonl'] if p['jsonl'] is not None else 'UNREADABLE'}")
        print(f"  Launch telemetry settings: {p['setting'] or 'none found in readable launch files'}")
    n70 = sum(p["domain"] == "70" for p in processes)
    if n70 > 1:
        print("\nSTOP: Multiple ROS domain 70 controller-manager processes. Confirm hardware ownership with lab operator.")
    elif any(p["domain"] in ("<unreadable>", "<unset>") for p in processes):
        print("\nWARNING: Some ROS domains unknown. Confirm hardware/controller ownership.")
    live = [p for p in processes if p["domain"] == "70" and p["isolated_loaded"] is True and p["jsonl"]]
    if live:
        print("RESULT: Isolated controller library and open raw JSONL seen; data quality/labels NOT verified.")
    else:
        print("RESULT: Active instrumented recorder NOT CONFIRMED; do not switch or restart controllers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
