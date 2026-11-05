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

"""Read-only structural preflight for the official LIBERO-10 Task 0 RECAP data."""

import argparse
import json
from pathlib import Path

DATASETS = (
    "libero10_task0_sft",
    "libero10_task0_train",
    "libero10_task0_eval",
)


def check_data_root(root: Path) -> list[str]:
    """Report missing/broken metadata; no decoding or dataset modifications."""
    problems = []
    if not root.is_dir():
        return [f"Dataset root is missing or not a directory: {root}"]

    for name in DATASETS:
        dataset = root / name
        if not dataset.is_dir():
            problems.append(f"{name}: missing dataset directory")
            continue

        info_path = dataset / "meta" / "info.json"
        if not info_path.is_file():
            problems.append(f"{name}: missing meta/info.json")
            continue

        try:
            with info_path.open(encoding="utf-8") as stream:
                info = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            problems.append(f"{name}: invalid meta/info.json ({error})")
            continue

        if not isinstance(info, dict):
            problems.append(f"{name}: meta/info.json must be a JSON object")
            continue

        for field in ("fps", "total_episodes", "features"):
            if field not in info:
                problems.append(f"{name}: missing metadata field {field}")

        if not (dataset / "data").is_dir():
            problems.append(f"{name}: missing data/ directory")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the three official LIBERO Task 0 datasets",
    )
    args = parser.parse_args()
    problems = check_data_root(args.data_root.expanduser())
    if problems:
        for issue in problems:
            print(f"FAIL: {issue}")
        return 1

    print("PASS: three dataset directories and basic LeRobot metadata found.")
    print("NOT CHECKED: videos, parquet rows, success labels, transforms, GPU, or training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
