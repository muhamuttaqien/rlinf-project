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

"""Collect or validate synthetic FR3 episodes without Ray or physical devices."""

import argparse
import json
import sys
from pathlib import Path

from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset
from rlinf.data.storage.fr3_synthetic import validate_synthetic_dataset
from toolkits.realworld_check.validate_fr3_config import load_config


def main() -> None:
    """Use only the fixed offline Hydra config and native local replay storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.validate_only and args.overrides:
        parser.error("Validation reads stored metadata and accepts no config overrides")
    report = (
        validate_synthetic_dataset(args.output)
        if args.validate_only
        else collect_synthetic_dataset(load_config(args.overrides), args.output)
    )
    sys.stdout.write(json.dumps(report, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
