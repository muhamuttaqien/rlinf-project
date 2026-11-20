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

"""Offline synthetic FR3 batching, loss and gradient validation; never training."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from rlinf.data.datasets.openpi.fr3 import FR3DatasetConfig, FR3OfflineDataset
from rlinf.data.datasets.openpi.fr3_checkpoint_audit import (
    audit_conversion_fixture,
    runtime_compatibility_report,
    verify_training_sources,
)
from rlinf.data.datasets.openpi.fr3_pipeline import (
    FR3TrainingConfig,
    FR3TrainingDataset,
    build_fr3_training_dataloader,
    synthetic_normalization_assets,
    synthetic_training_fixture,
)
from rlinf.data.datasets.openpi.fr3_training import inspect_checkpoint_assets
from rlinf.data.storage.fr3_synthetic import GATE, write_json
from rlinf.data.storage.fr3_validation import require
from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.models.embodiment.openpi.rlt_config import OpenPiPytorchRLTConfig
from rlinf.models.embodiment.openpi.sft_loss import (
    masked_micro_batch_count,
    normalize_masked_micro_batches,
)


class SyntheticLossProbe(Pi0):
    """A CPU scalar-output fixture using native SFT handling, without a VLA network."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.probe = torch.nn.Parameter(torch.zeros(32))
        self.action_dim = 32
        self.action_chunk = self.action_env_dim = None
        self.rlt_cfg = OpenPiPytorchRLTConfig()
        self._offline_fr3_loss_probe = True

    def compute_loss(
        self, observation: Any, actions: torch.Tensor, *, train: bool = False
    ) -> torch.Tensor:
        """Evaluate deterministic per-element MSE for the isolated tensor fixture."""
        del observation, train
        return (self.probe[None, None, :] - actions).square()

    def gradient_checkpointing_disable(self, **kwargs: Any) -> None:
        """The scalar fixture has no recomputed layers."""


def validate_masked_probe(batches: list[dict]) -> dict:
    """Check native SFT forward and accumulated gradients against one global mean."""
    count = sum(masked_micro_batch_count(batch) for batch in batches)
    normalized = normalize_masked_micro_batches(
        batches, world_size=1, global_valid_count=count
    )
    model = SyntheticLossProbe()
    observed = 0.0
    for batch in normalized:
        loss = model.sft_forward(batch)
        observed += loss.item() / len(normalized)
        (loss / len(normalized)).backward()
    actions = torch.cat([b["actions"] for b in batches])
    mask = torch.cat([b["action_loss_mask"] for b in batches])
    expected = actions.square().masked_select(mask).sum() / count
    expected_gradient = (
        -2 * torch.where(mask, actions, torch.zeros_like(actions))
    ).sum(dim=(0, 1)) / count
    require(
        abs(observed - expected.item()) < 1e-6
        and torch.allclose(model.probe.grad, expected_gradient, atol=1e-7),
        "Masked accumulated loss/gradient disagrees with global element mean",
    )
    changed = copy.copy(normalized[0])
    changed["actions"] = torch.where(
        changed["action_loss_mask"],
        changed["actions"],
        torch.full_like(changed["actions"], 1e6),
    )
    require(
        torch.equal(model.sft_forward(changed), model.sft_forward(normalized[0])),
        "Padding affected native masked loss",
    )
    require(not model.probe.grad[7:].any(), "Padded channels contributed gradients")
    return {
        "status": "PASS",
        "loss": observed,
        "expected_loss": expected.item(),
        "valid_elements": count,
        "micro_batches": len(batches),
        "padded_channel_gradient_max": model.probe.grad[7:].abs().max().item(),
        "padding_value_invariance": True,
        "optimizer_created": False,
        "full_model_constructed": False,
        "checkpoint_eligible": False,
    }


def validate_training_pipeline(
    export: str | Path, assets: str | Path, output: str | Path
) -> dict:
    """Emit independent structural, checkpoint, dataset and physical decisions."""
    names = [
        "native_dataset_validity",
        "fr3_training_schema",
        "action_target_structure",
        "action_target_semantics",
        "state_preprocessing",
        "camera_preprocessing",
        "task_instructions",
        "episode_split",
        "training_normalization",
        "loss_mask_integration",
        "checkpoint_source",
        "converted_weights",
        "model_runtime",
        "checkpoint_execution",
        "recap_experiments",
        "physical_execution",
    ]
    report = {
        "schema": "fr3_training_readiness_v1",
        **GATE,
        "offline_smoke_complete": False,
        "training_enabled": False,
        "decisions": {n: {"status": "UNQUALIFIED"} for n in names},
        "diagnostics": [],
    }
    report["decisions"]["physical_execution"] = {
        "status": "UNQUALIFIED",
        "authorization": "NO_GO / NOT_GRANTED",
    }
    report["decisions"]["converted_weights"] = {
        "status": "NOT_EXECUTED",
        "reason": "WEIGHT_CONVERSION_NOT_EXECUTED",
    }
    try:
        profile = verify_training_sources()
        report["training_contract"] = profile
        report["checkpoint_small_assets"] = inspect_checkpoint_assets(assets)
        report["runtime"] = runtime_compatibility_report()
        windows = []
        for horizon in (1, 4, 20, 50):
            native = FR3OfflineDataset(
                export, config=FR3DatasetConfig(action_horizon=horizon)
            )
            fixture = synthetic_training_fixture(native)
            statistics = synthetic_normalization_assets(native, fixture)
            dataset = FR3TrainingDataset(
                native,
                config=FR3TrainingConfig(assets),
                fixture=fixture,
                normalization=statistics,
            )
            loader = build_fr3_training_dataloader(dataset, batch_size=2)
            batches = list(loader)
            result = validate_masked_probe(batches)
            windows.append(
                {
                    "horizon": horizon,
                    "policy": "masked_padded_windows",
                    "samples": len(dataset),
                    "padded_future_steps": sum(
                        int((~b["valid_future_steps"]).sum()) for b in batches
                    ),
                    "cross_episode_windows": 0,
                    "loss_probe": result,
                }
            )
            if horizon == 20:
                batch = batches[0]
                report["batch"] = {
                    "state": list(batch["observation"].state.shape),
                    "native_state": list(batch["native_state"].shape),
                    "actions": list(batch["actions"].shape),
                    "mask": list(batch["action_loss_mask"].shape),
                    "images": {
                        k: list(v.shape) for k, v in batch["observation"].images.items()
                    },
                    "prompt_tokens": list(batch["observation"].tokenized_prompt.shape),
                    "scope": "synthetic_math_only",
                }
                report["dataset"] = {
                    "dataset_id": native.manifest["dataset_id"],
                    "source_manifest_sha256": native.source_manifest_sha256,
                    "episodes": native.audit["episodes"],
                    "transitions": len(native),
                    "synthetic_training_frames": len(dataset),
                }
        report["windows"] = windows
        report["conversion"] = audit_conversion_fixture(
            Path(output) / "synthetic_conversion_audit"
        )
        for name in (
            "native_dataset_validity",
            "fr3_training_schema",
            "action_target_structure",
            "loss_mask_integration",
            "checkpoint_source",
        ):
            report["decisions"][name] = {
                "status": "PASS",
                "scope": "offline_synthetic_software_only",
            }
        report["decisions"]["state_preprocessing"]["reason"] = (
            "42D fixture-normalized structure passes; checkpoint semantics unavailable"
        )
        report["decisions"]["camera_preprocessing"]["reason"] = (
            "Synthetic resize/masks pass; physical camera roles/calibration unavailable"
        )
        report["decisions"]["task_instructions"]["reason"] = (
            "Only explicitly attributed synthetic fixture text exists"
        )
        report["decisions"]["episode_split"]["reason"] = (
            "Test-only partition; no approved real training split"
        )
        report["decisions"]["training_normalization"]["reason"] = (
            "Constant synthetic math fixtures; production qualification disabled"
        )
        report["decisions"]["model_runtime"]["reason"] = (
            "Isolated environment required; dependency/ownership conflicts"
        )
        report["offline_smoke_complete"] = True
    except Exception as exc:
        report["diagnostics"].append({"type": type(exc).__name__, "message": str(exc)})
        report["decisions"]["fr3_training_schema"] = {"status": "FAIL"}
    return report


def main() -> int:
    """Run offline probes into a new output root separate from every input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        for source in (args.export.resolve(), args.assets.resolve()):
            require(
                output != source
                and output not in source.parents
                and source not in output.parents,
                "Output must be separate from dataset and asset roots",
            )
        output.mkdir(parents=True, exist_ok=False)
        report = validate_training_pipeline(args.export, args.assets, output)
        write_json(output / "readiness.json", report)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "offline_smoke_complete",
                        "training_enabled",
                        "decisions",
                        "diagnostics",
                    )
                }
            )
        )
        return 0 if report["offline_smoke_complete"] else 1
    except (OSError, ValueError) as exc:
        print(json.dumps({"offline_smoke_complete": False, "error": str(exc), **GATE}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
