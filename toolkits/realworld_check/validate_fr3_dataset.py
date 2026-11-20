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

"""Offline FR3 dataset assessment, optional native LeRobot export and reload audit."""

import argparse
import json
from pathlib import Path

from rlinf.data.storage.fr3_synthetic import GATE, write_json
from rlinf.data.storage.fr3_validation import assess_synthetic_dataset, require


def main() -> int:
    """Write assessment to a new separate directory; return nonzero on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--export", action="store_true", help="Export through the native LeRobot writer"
    )
    parser.add_argument(
        "--reload-export",
        type=Path,
        help="Independently audit an existing export against this dataset",
    )
    args = parser.parse_args()
    root, output = args.dataset.resolve(), args.output.resolve()
    try:
        require(
            output != root
            and root not in output.parents
            and output not in root.parents,
            "Assessment output must be separate from source dataset",
        )
        require(
            not (args.export and args.reload_export),
            "Select either export or reload audit",
        )
        output.mkdir(parents=True, exist_ok=False)
        report = assess_synthetic_dataset(root)
        if (args.export or args.reload_export) and report["valid"]:
            try:
                from rlinf.data.storage.lerobot.fr3_export import (
                    audit_synthetic_export,
                    export_synthetic_dataset,
                )

                if args.export:
                    report["export"] = export_synthetic_dataset(root, output / "export")
                else:
                    exported = json.loads(
                        (args.reload_export / "export.json").read_text()
                    )
                    require(
                        exported["source_manifest_sha256"]
                        == report["source_manifest_sha256"],
                        "Reload export belongs to a different source manifest",
                    )
                    report["export"] = {
                        **exported,
                        "audit": audit_synthetic_export(args.reload_export),
                    }
                report["readiness"]["lerobot_export"] = (
                    "PASS_COMPATIBLE_SUBSET_WITH_NATIVE_ARCHIVE"
                )
            except Exception as exc:
                report["export"] = {
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                report["readiness"]["lerobot_export"] = "FAIL"
        elif args.export or args.reload_export:
            report["export"] = {"status": "BLOCKED_INVALID_DATASET"}
            report["readiness"]["lerobot_export"] = "BLOCKED_INVALID_DATASET"
        write_json(output / "quality.json", report)
        success = report["valid"] and report["export"]["status"] != "FAILED"
        print(
            json.dumps(
                {
                    "valid": report["valid"],
                    "counts": report["counts"],
                    "readiness": report["readiness"],
                    **GATE,
                },
                sort_keys=True,
            )
        )
        return 0 if success else 1
    except (ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "valid": False,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    **GATE,
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
