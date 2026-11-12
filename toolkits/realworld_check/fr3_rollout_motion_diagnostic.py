"""Read-only comparison of recorded pi0.5 arm commands and observed FR3 TCP motion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def diagnose(command_log: Path) -> dict:
    records = [json.loads(line) for line in command_log.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("No recorded ROS commands")
    xyz = [tuple(float(c) for c in record["tcp_xyz"]) for record in records]
    if any(len(point) != 3 or not all(math.isfinite(v) for v in point) for point in xyz):
        raise ValueError("Invalid measured TCP positions")
    arm = [record["sent_arm_6d"] for record in records]
    if any(len(row) != 6 or not all(math.isfinite(float(x)) for x in row) for row in arm):
        raise ValueError("Invalid arm request values")
    requested = [record["requested_7d"][:6] for record in records]
    return {
        "published_arm_requests": len(records),
        "max_absolute_arm_request": round(max(abs(float(x)) for row in arm for x in row), 5),
        "max_absolute_pi05_arm_prediction": round(max(abs(float(x)) for row in requested for x in row), 5),
        "max_tcp_displacement_mm": round(max(math.dist(xyz[0], point) for point in xyz) * 1000, 3),
        "final_tcp_displacement_mm": round(math.dist(xyz[0], xyz[-1]) * 1000, 3),
        "tcp_path_length_mm": round(sum(math.dist(a, b) for a, b in zip(xyz, xyz[1:])) * 1000, 3),
        "tcp_axis_span_mm": [round((max(p[i] for p in xyz) - min(p[i] for p in xyz)) * 1000, 3) for i in range(3)],
        "logged_gripper_events": {
            event: sum(r.get("gripper_event") == event for r in records)
            for event in ("open", "close", "hold")
        },
        "note": "ROS publish events are not proof that the controller accepted each request.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.episode / "policy_command_log.jsonl"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
