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

"""Qualify pinned FR3 checkpoint assets offline without constructing a model."""

import argparse
import json
from pathlib import Path

import numpy as np

from rlinf.data.datasets.openpi.fr3 import FR3DatasetConfig, FR3OfflineDataset
from rlinf.data.datasets.openpi.fr3_training import (
    assess_fr3_training_requirements,
    build_fr3_request_target_proposal,
    inspect_checkpoint_assets,
    load_checkpoint_candidate,
    require_fr3_training_ready,
    tokenize_fr3_fixture,
    verify_checkpoint_sources,
)
from rlinf.data.storage.fr3_synthetic import GATE, write_json
from rlinf.data.storage.fr3_validation import require


def qualify_fr3_checkpoint(export_path: str | Path, assets_root: str | Path) -> dict:
    """Check candidate identity, asset schemas, dataset windows and batch containers."""
    report = {
        "schema": "fr3_checkpoint_qualification_v1",
        **GATE,
        "offline_inspection_complete": False,
        "diagnostics": [],
        "training_run": False,
        "rollouts_run": False,
        "model_constructed": False,
        "decisions": {
            "structural_compatibility": "NOT_CHECKED",
            "checkpoint_compatibility": "CHECKPOINT_UNVERIFIED",
            "training_data_readiness": "UNQUALIFIED_SYNTHETIC_ONLY",
            "recap_readiness": "UNQUALIFIED_NO_EXPERIMENTAL_ANNOTATIONS",
            "physical_authorization": "NO_GO / NOT_GRANTED",
        },
    }
    try:
        candidate = load_checkpoint_candidate()
        report["candidate"] = candidate
        report["runtime"] = verify_checkpoint_sources(candidate)
        report["assets"] = inspect_checkpoint_assets(assets_root)
        horizon = candidate["configuration"]["inspection_horizon"]
        dataset = FR3OfflineDataset(
            export_path, config=FR3DatasetConfig(action_horizon=horizon)
        )
        report["dataset"] = {
            "dataset_id": dataset.manifest["dataset_id"],
            "source_manifest_sha256": dataset.source_manifest_sha256,
            "episodes": dataset.audit["episodes"],
            "transitions": len(dataset),
            "validity": "PASS_NATIVE_ARCHIVE_AND_LEROBOT_AUDIT",
        }
        report["training_metadata"] = assess_fr3_training_requirements(dataset)
        valid_elements = padded_steps = full_windows = 0
        for index in range(len(dataset)):
            proposal = build_fr3_request_target_proposal(dataset, index)
            mask = proposal["proposed_loss_mask"]
            valid_elements += int(mask.sum())
            padded_steps += int((~mask[:, :7].any(axis=1)).sum())
            full_windows += int(mask[:, :7].all())
            require(
                proposal["model_training_target"] is None,
                "Training target guard changed",
            )
        report["temporal"] = {
            "inspection_horizon": horizon,
            "pretraining_horizon": None,
            "period_ns": dataset.manifest["step_period_ns"],
            "full_windows": full_windows,
            "padded_future_steps": padded_steps,
            "proposed_valid_loss_elements": valid_elements,
            "cross_episode_windows": 0,
            "native_sft_consumes_padding_mask": False,
            "qualification": "BLOCKED_NATIVE_PER_SAMPLE_LOSS_MASK_REQUIRED",
        }
        tokenization = tokenize_fr3_fixture(
            assets_root,
            np.zeros(42, dtype=np.float32),
            "synthetic tokenizer schema probe only",
            scope="synthetic_math_only",
        )
        report["tokenizer"] = {
            k: v for k, v in tokenization.items() if k not in ("tokens", "mask")
        }
        report["tokenizer"].update(
            token_shape=list(tokenization["tokens"].shape),
            dtype=str(tokenization["tokens"].dtype),
            authentic_instruction_available=False,
            normalized_native_state_available=False,
        )
        import torch
        from openpi.models.pi0_config import Pi0Config

        from rlinf.models.embodiment.openpi.modules.model import Observation

        sample = dataset[0]
        observation = Observation.from_dict(
            {
                "state": torch.from_numpy(sample["native_state"][None].copy()),
                "image": {
                    k: torch.from_numpy(v[None].copy())
                    for k, v in sample["image"].items()
                },
                "image_mask": {
                    k: torch.from_numpy(v.reshape(1).copy())
                    for k, v in sample["image_mask"].items()
                },
            }
        )
        spec, action_spec = Pi0Config(pi05=True, action_horizon=horizon).inputs_spec(
            batch_size=1
        )
        first_proposal = build_fr3_request_target_proposal(dataset, 0)
        proposed_actions = torch.from_numpy(
            first_proposal["proposed_actions"][None].copy()
        )
        require(
            tuple(proposed_actions.shape) == action_spec.shape,
            "Proposed action batch disagrees with declared model action shape",
        )
        report["batch_structure"] = {
            "container": "RLinf_Observation",
            "native_state_shape": list(observation.state.shape),
            "declared_default_state_shape": list(spec.state.shape),
            "state_matches_default_spec": observation.state.shape == spec.state.shape,
            "proposed_action_shape": list(proposed_actions.shape),
            "declared_action_shape": list(action_spec.shape),
            "image_shapes": {k: list(v.shape) for k, v in observation.images.items()},
            "tokenizer_fixture_is_native_training_prompt": False,
            "strict_checkpoint_batch_qualified": False,
            "pi05_state_context": "discrete_tokens_before_padding_continuous_state_unused",
        }
        try:
            require_fr3_training_ready()
        except ValueError as exc:
            report["training_execution_rejection"] = str(exc)
        else:
            raise ValueError("Expected checkpoint/training rejection")
        report["decisions"]["structural_compatibility"] = "PASS_OFFLINE_PROBES_ONLY"
        report["offline_inspection_complete"] = True
    except Exception as exc:
        report["diagnostics"].append({"type": type(exc).__name__, "message": str(exc)})
        report["decisions"]["structural_compatibility"] = "FAIL_INSPECTION"
    return report


def main() -> int:
    """Write a non-destructive report in a new directory outside all input roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        for source in (args.export.resolve(), args.assets.resolve()):
            require(
                output != source
                and output not in source.parents
                and source not in output.parents,
                "Qualification output must be separate from dataset and asset inputs",
            )
        output.mkdir(parents=True, exist_ok=False)
        report = qualify_fr3_checkpoint(args.export, args.assets)
        write_json(output / "qualification.json", report)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("offline_inspection_complete", "decisions", "diagnostics")
                }
            )
        )
        return 0 if report["offline_inspection_complete"] else 1
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {"offline_inspection_complete": False, "error": str(exc), **GATE}
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
