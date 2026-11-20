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

"""Offline structural FR3/OpenPI and RECAP qualification, without model execution."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rlinf.data.datasets.openpi.fr3 import FR3DatasetConfig, FR3OfflineDataset
from rlinf.data.datasets.openpi.fr3_contract import (
    CONTRACT_ID,
    CONTRACT_PATH,
    CONTRACT_SHA256,
    normalization_fixture_check,
)
from rlinf.data.storage.fr3_synthetic import GATE, write_json
from rlinf.data.storage.fr3_validation import require


def qualify_fr3_dataset(
    export_path: str | Path,
    *,
    config: FR3DatasetConfig | None = None,
    annotations_path: str | Path | None = None,
    check_normalization_fixture: bool = False,
) -> dict:
    """Audit structural samples and return separate checkpoint/physical decisions."""
    report = {
        "schema": "fr3_openpi_recap_qualification_v1",
        **GATE,
        "contract_id": CONTRACT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "structural_validation_complete": False,
        "diagnostics": [],
        "decisions": {
            "dataset_validity": "NOT_CHECKED",
            "structural_openpi_compatibility": "NOT_CHECKED",
            "semantic_state_compatibility": "UNQUALIFIED_CHECKPOINT",
            "semantic_action_compatibility": "UNQUALIFIED_MODEL_TARGET",
            "camera_compatibility": "UNQUALIFIED_PHYSICAL_ROLES",
            "normalization_qualification": "UNQUALIFIED_NO_CHECKPOINT_ASSETS",
            "checkpoint_compatibility": "CHECKPOINT_UNVERIFIED",
            "recap_metadata_readiness": "NOT_CHECKED",
            "physical_execution_authorization": "NO_GO / NOT_GRANTED",
        },
    }
    try:
        dataset = FR3OfflineDataset(
            export_path, config=config, annotations_path=annotations_path
        )
        padded_windows = padded_elements = images_checked = 0
        selected_episodes = set()
        for index in range(len(dataset)):
            sample = dataset[index]
            selected_episodes.add(sample["provenance"]["episode_id"])
            require(
                sample["model_training_target"] is None
                and sample["prompt"] is None
                and sample["physical_execution"] is False,
                "Unqualified model target, prompt or execution claim",
            )
            padded_windows += int(sample["actions_is_pad"].any())
            padded_elements += int(sample["actions_is_pad"].sum())
            for name, image in sample["raw_images"].items():
                expected = sample["provenance"]["observation"]["cameras"][name][
                    "sha256"
                ]
                require(
                    hashlib.sha256(image.tobytes()).hexdigest() == expected,
                    f"FR3 camera integrity mismatch: {name}",
                )
                images_checked += 1
        report.update(
            runtime=dataset.runtime,
            official_reference=dataset.contract["official_reference"],
            reference_configurations=dataset.contract["openpi"]["configs"],
            checkpoint=dataset.contract["checkpoint"],
            dataset_id=dataset.manifest["dataset_id"],
            source_manifest_sha256=dataset.source_manifest_sha256,
            source_episodes=dataset.audit["episodes"],
            source_transitions=dataset.audit["transitions"],
            episodes=len(selected_episodes),
            transitions=len(dataset),
            split_selection=dataset.config.split,
            dimensions={
                "native_state": 42,
                "structural_state": 42,
                "checkpoint_state": None,
                "recorded_action_window": [dataset.config.action_horizon, 7],
                "model_training_target": None,
                "probe_images": [224, 224, 3],
            },
            windows={
                "policy": dataset.config.padding_policy,
                "horizon": dataset.config.action_horizon,
                "padded_windows": padded_windows,
                "padded_elements": padded_elements,
                "cross_episode_samples": 0,
            },
            camera_images_verified=images_checked,
            camera_slots=dataset.contract["cameras"],
            recap=dataset.metadata_report,
            metadata=dataset.metadata_document(),
            normalization_fixture={
                "status": "NOT_REQUESTED",
                "checkpoint_qualification": "UNQUALIFIED",
            },
        )
        if check_normalization_fixture:
            fixture = {
                "schema": "fr3_normalization_fixture_v1",
                "scope": "synthetic_math_only",
                "contract_id": CONTRACT_ID,
                "method": "zscore",
                "statistics": {
                    k: {"mean": [0.0] * n, "std": [1.0] * n}
                    for k, n in (("state", 42), ("actions", 7))
                },
            }
            check = normalization_fixture_check(
                np.zeros(42, dtype=np.float32),
                np.zeros((dataset.config.action_horizon, 7), dtype=np.float32),
                fixture,
            )
            report["normalization_fixture"] = {
                k: v for k, v in check.items() if k not in ("normalized", "restored")
            }
        try:
            dataset.require_model_ready()
        except ValueError as exc:
            report["model_execution_rejection"] = str(exc)
        else:
            raise ValueError("Expected checkpoint execution to remain blocked")
        report["decisions"].update(
            dataset_validity="PASS",
            structural_openpi_compatibility="PASS_OBSERVATION_CONTAINER_ONLY",
            camera_compatibility="PASS_SYNTHETIC_PROBE_PHYSICAL_ROLES_UNQUALIFIED",
            recap_metadata_readiness="PASS_SCHEMA_EXPERIMENTS_UNQUALIFIED",
        )
        report["structural_validation_complete"] = True
    except Exception as exc:
        report["diagnostics"].append({"type": type(exc).__name__, "message": str(exc)})
        report["decisions"]["structural_openpi_compatibility"] = "FAIL"
    return report


def main() -> int:
    """Write a quality report to a new directory separated from the source bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--split", choices=("unassigned", "train", "validation", "test")
    )
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--padding-policy", default="lerobot_repeat_last_with_mask_v1")
    parser.add_argument("--check-normalization-fixture", action="store_true")
    args = parser.parse_args()
    try:
        source, output = args.export.resolve(), args.output.resolve()
        require(
            source != output
            and source not in output.parents
            and output not in source.parents,
            "Qualification output must be separate from source bundle",
        )
        output.mkdir(parents=True, exist_ok=False)
        report = qualify_fr3_dataset(
            source,
            config=FR3DatasetConfig(
                args.contract, args.horizon, args.padding_policy, args.split
            ),
            annotations_path=args.annotations,
            check_normalization_fixture=args.check_normalization_fixture,
        )
        write_json(output / "qualification.json", report)
        print(
            json.dumps(
                {
                    "structural_validation_complete": report[
                        "structural_validation_complete"
                    ],
                    "decisions": report["decisions"],
                    "diagnostics": report["diagnostics"],
                    **GATE,
                },
                sort_keys=True,
            )
        )
        return 0 if report["structural_validation_complete"] else 1
    except (ValueError, OSError) as exc:
        print(
            json.dumps(
                {"structural_validation_complete": False, "error": str(exc), **GATE}
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
