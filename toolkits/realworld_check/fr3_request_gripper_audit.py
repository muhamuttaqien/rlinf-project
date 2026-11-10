# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Read-only analysis of gripper REQUEST vs measured knuckle angle.

Does not infer controller accepted/executed targets or calibrate the
Robotiq bridge; no ROS, no robot commands, and no training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def audit(candidate_root: Path) -> dict:
    root = candidate_root.expanduser().resolve()
    source = json.loads((root / "report.json").read_text())
    if source.get("schema") != "fr3_spacemouse_request_alignment_candidate_v1":
        raise ValueError("Expected audited SpaceMouse candidate dataset")
    positions = defaultdict(list)
    changes = 0
    episodes = 0
    unknown = 0
    for ep in source["episode_reports"]:
        episodes += 1
        path = root / "episodes" / (ep["episode"] + ".jsonl")
        last = None
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                request = row.get("gripper_request_open_fraction")
                if request is None:
                    unknown += 1
                    continue
                if request not in (0, 1):
                    raise ValueError("Nonbinary gripper button request")
                state = row["state19_candidate"]
                if len(state) != 19:
                    raise ValueError("Invalid candidate state shape")
                positions[str(int(request))].append(float(state[17]))
                if last is not None and last != request:
                    changes += 1
                last = request
    return {
        "episodes": episodes,
        "unknown_prefix_rows": unknown,
        "request_open_1_knuckle_median_rad": median(positions["1"]) if positions["1"] else None,
        "request_close_0_knuckle_median_rad": median(positions["0"]) if positions["0"] else None,
        "rows_by_request": {k: len(v) for k, v in positions.items()},
        "recorded_request_transitions": changes,
        "note": (
            "Descriptive association only: commanded gripper is a sparse "
            "latched button request. Measured knuckle may lag, retain a "
            "previous episode's state, and be affected by contact. "
            "Neither target direction nor controller acceptance is verified."
        ),
        "controller_mapping_verified": False,
        "cross_host_timing_verified": False,
        "robot_execution_approved": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.candidate_root), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
