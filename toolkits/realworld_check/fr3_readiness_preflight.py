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

"""Report passive FR3/Robotiq evidence; physical authorization is never granted."""

import argparse
import json
import sys
from pathlib import Path

from rlinf.robotics.parts.ros2_readiness import run_preflight


def main() -> None:
    """Write structured JSON; exit 1 for failed evidence, 2 for unknown evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--max-age", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--ros-setup", default="/opt/ros/jazzy/setup.bash")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--nuc",
        action="store_true",
        help="Also run the fixed read-only Humble probe through authorized SSH alias nuc-franka",
    )
    args = parser.parse_args()
    try:
        report = run_preflight(
            **{key: value for key, value in vars(args).items() if key != "output"}
        )
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    states = [
        value["state"]
        for key, value in report["categories"].items()
        if key != "physical_safety_qualification"
    ]
    raise SystemExit(1 if "FAIL" in states else 2 if "UNKNOWN" in states else 0)


if __name__ == "__main__":
    main()
