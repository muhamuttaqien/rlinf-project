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

"""Bounded, subscription-only FR3/Robotiq observation check."""

import argparse
import json
import math
import time

from rlinf.robotics import FrankaRobot
from rlinf.robotics.parts.ros2_observation import (
    FR3ROS2Observation,
    RobotiqROS2Observation,
)
from rlinf.utils.logging import get_logger


def main() -> None:
    """Attach to existing state publishers and report completeness and age."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--max-age", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--ros-setup", default="/opt/ros/jazzy/setup.bash")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("Expected positive finite --duration")
    settings = {
        "max_age": args.max_age,
        "ros_python": args.ros_python,
        "ros_setup": args.ros_setup,
    }
    robot = FrankaRobot(
        arm=FR3ROS2Observation(domain_id=70, **settings),
        end_effector=RobotiqROS2Observation(domain_id=71, **settings),
    )
    report = {
        name: {
            "reads": 0,
            "incomplete": 0,
            "max_receipt_age_s": {},
            "frames": {},
            "last_status": None,
        }
        for name in robot.children
    }
    logger = get_logger()
    try:
        robot.connect()
        for part in robot.children.values():
            part.wait_for_observation(args.timeout)
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            for name, observation in robot.get_observation().items():
                status = observation["ros2"]
                record = report[name]
                record["reads"] += 1
                record["incomplete"] += not status["complete"]
                record["last_status"] = status
                for key, stream in status["streams"].items():
                    if stream["sample"] is not None:
                        record["frames"][key] = stream["sample"]["header"]["frame_id"]
                        record["max_receipt_age_s"][key] = max(
                            record["max_receipt_age_s"].get(key, 0), stream["age_s"]
                        )
            time.sleep(0.02)
    finally:
        robot.disconnect()
    logger.info(json.dumps(report, allow_nan=False))
    if any(record["incomplete"] for record in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
