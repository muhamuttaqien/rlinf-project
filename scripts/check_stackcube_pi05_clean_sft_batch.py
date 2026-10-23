# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify ONE actual OpenPI SFT loader batch against LeRobot frame 0.

Uses the same RLinf -> official OpenPI loader dispatch as the FSDP SFT worker.
Only the sampler (shuffle=False), worker count (0), and microbatch size (1)
are changed so that batch item 0 maps deterministically to raw dataset row 0.
All data/model transforms, norm stats, and action horizon are unchanged.

No model weights are loaded; no dataset/checkpoint/training files are modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from scripts.check_stackcube_pi05_clean_transforms import as_numpy, raw_sample


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_clean_v1.yaml"
MODEL_CONFIG = ROOT / "examples/sft/config/model/pi0_5.yaml"
EXPECTED_REPO = "local/stackcube_pi05_clean_v1"
EXPECTED_CONFIG = "pi05_stackcube_panda_easy_ee_delta_two_cam_state"


def make_check_config(
    config_path: Path, repo_id: str, stats_path: Path, model_path: Path,
):
    """Resolve the real SFT config, overriding only deterministic read settings."""
    cfg = OmegaConf.merge(
        {"actor": {"model": OmegaConf.load(MODEL_CONFIG)}},
        OmegaConf.load(config_path),
    )
    if repo_id != EXPECTED_REPO:
        raise ValueError(f"Expected isolated Clean V1 dataset {EXPECTED_REPO}, got {repo_id}")
    if str(cfg.actor.model.openpi.config_name) != EXPECTED_CONFIG:
        raise ValueError("Selected YAML is not the Clean V1 EE-delta config")
    if bool(cfg.actor.model.openpi.train_image_augmentation):
        raise ValueError("Clean V1 must disable train-only augmentation")
    if int(cfg.actor.model.openpi.action_horizon) != 10:
        raise ValueError("Clean V1 action horizon is not 10")

    cfg.data.train_data_paths = repo_id
    cfg.actor.model.model_path = str(model_path.resolve())
    cfg.actor.model.openpi_data.norm_stats_path = str(stats_path.resolve())
    cfg.actor.micro_batch_size = 1
    cfg.data.num_workers = 0
    return cfg


def expected_observation(expected: dict):
    """Apply upstream Observation.from_dict to the independent transformed row.

    DataLoaderImpl converts batched torch uint8 images to model-ready tensors
    (sometimes HWC -> CHW and [0,255] -> [-1,1]) while packaging them into
    Observation. Comparing the raw transform dictionary would be a false
    mismatch even when the data loader is entirely correct.
    """
    import torch
    from openpi.models.model import Observation

    def batch_tensor(value):
        return torch.as_tensor(np.expand_dims(as_numpy(value), axis=0).copy())

    converted = {
        "image": {key: batch_tensor(value) for key, value in expected["image"].items()},
        "image_mask": {
            key: batch_tensor(value)
            for key, value in expected["image_mask"].items()
        },
        "state": batch_tensor(expected["state"]),
    }
    for field in (
        "tokenized_prompt", "tokenized_prompt_mask",
        "token_ar_mask", "token_loss_mask",
    ):
        if field in expected:
            converted[field] = batch_tensor(expected[field])
    return Observation.from_dict(converted)


def compare_first_batch(observation, model_actions, expected: dict, raw: dict) -> dict:
    """Check that *loader* output equals independently transformed frame zero."""
    prepared = expected_observation(expected)
    model_actions = as_numpy(model_actions)
    if model_actions.shape != (1, 10, 32):
        raise AssertionError(f"Actual SFT target shape must be [1,10,32]: {model_actions.shape}")

    raw_actions = as_numpy(raw["actions"])
    if raw_actions.shape != (10, 7):
        raise AssertionError(f"Raw first action window must be [10,7]: {raw_actions.shape}")
    if as_numpy(raw["state"]).shape != (8,):
        raise AssertionError("Raw state is not 8D")

    np.testing.assert_allclose(
        model_actions[0], as_numpy(expected["actions"]),
        atol=1e-6, rtol=0, err_msg="Actual loader action targets differ from row 0",
    )
    np.testing.assert_allclose(
        as_numpy(observation.state), as_numpy(prepared.state),
        atol=1e-6, rtol=0, err_msg="Actual loader state differs from row 0",
    )

    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        if key not in observation.images or key not in observation.image_masks:
            raise AssertionError(f"Loader missing image or mask for {key}")
        np.testing.assert_allclose(
            as_numpy(observation.images[key]),
            as_numpy(prepared.images[key]),
            atol=1e-6, rtol=0, err_msg=f"Actual SFT image mismatch: {key}",
        )
        np.testing.assert_array_equal(
            as_numpy(observation.image_masks[key]),
            as_numpy(prepared.image_masks[key]),
            err_msg=f"Actual SFT camera mask mismatch: {key}",
        )
    for key in ("base_0_rgb", "left_wrist_0_rgb"):
        if not bool(as_numpy(observation.image_masks[key])[0]):
            raise AssertionError(f"Real camera {key} was masked in SFT batch")
    if bool(as_numpy(observation.image_masks["right_wrist_0_rgb"])[0]):
        raise AssertionError("Third (placeholder) camera must be masked")

    for field in ("tokenized_prompt", "tokenized_prompt_mask"):
        actual_tokens = getattr(observation, field)
        expected_tokens = getattr(prepared, field)
        if actual_tokens is None or expected_tokens is None:
            raise AssertionError(f"Required OpenPI prompt field absent: {field}")
        np.testing.assert_array_equal(
            as_numpy(actual_tokens), as_numpy(expected_tokens),
            err_msg=f"Actual SFT token mismatch: {field}",
        )

    return {
        "batch_action_shape": list(model_actions.shape),
        "raw_action_shape": list(raw_actions.shape),
        "raw_state_dim": 8,
        "train_images": 2,
        "first_raw_action": raw_actions[0].tolist(),
        "first_normalized_32d_action": model_actions[0, 0].tolist(),
    }


def run(config_path: Path, repo_id: str, stats_path: Path, model_path: Path) -> dict:
    from openpi import transforms
    from rlinf.data.datasets.openpi import build_openpi_sft_dataloader

    cfg = make_check_config(config_path, repo_id, stats_path, model_path)

    # Exactly the worker's entrypoint, but eval_dataset=True disables shuffle.
    # This preserves its actual transform/data-config path, unlike manually
    # constructing a TransformedDataset and calling dataset[0].
    data_loader, data_cfg = build_openpi_sft_dataloader(
        cfg, world_size=1, rank=0, data_paths=cfg.data.train_data_paths,
        eval_dataset=True,
    )
    if data_cfg.repo_id != repo_id or data_cfg.norm_stats is None:
        raise AssertionError("Actual OpenPI loader did not select clean dataset/stats")

    actual_observation, actual_actions = next(iter(data_loader))
    raw = raw_sample(repo_id)
    full_input = transforms.compose([
        *data_cfg.repack_transforms.inputs,
        *data_cfg.data_transforms.inputs,
        transforms.Normalize(
            data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm,
        ),
        *data_cfg.model_transforms.inputs,
    ])
    expected = full_input(raw)
    report = compare_first_batch(actual_observation, actual_actions, expected, raw)

    # Independent check: decode the ACTUAL loader batch (not expected actions)
    # through the data_config's own inverse transforms back to the raw 7D target.
    decode = transforms.compose([
        *data_cfg.model_transforms.outputs,
        transforms.Unnormalize(
            data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm,
        ),
        *data_cfg.data_transforms.outputs,
    ])
    decoded = as_numpy(decode({
        "actions": as_numpy(actual_actions)[0],
        "state": as_numpy(actual_observation.state)[0],
    })["actions"])
    if decoded.shape != (10, 7):
        raise AssertionError(f"Decoded loader target must be [10,7]: {decoded.shape}")
    np.testing.assert_allclose(
        decoded, as_numpy(raw["actions"]), atol=1e-4, rtol=1e-4,
        err_msg="Decoded ACTUAL loader actions diverge from persisted raw actions",
    )
    report.update({
        "dataset": repo_id,
        "model_config": str(cfg.actor.model.openpi.config_name),
        "decoded_action_shape": list(decoded.shape),
        "first_decoded_action": decoded[0].tolist(),
        "status": "PASSED",
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-id", default=EXPECTED_REPO)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    if not args.norm_stats.expanduser().is_file():
        raise FileNotFoundError(f"Missing norm_stats.json: {args.norm_stats}")
    if not args.model_path.expanduser().is_dir():
        raise FileNotFoundError(f"Missing original π0.5 model dir: {args.model_path}")
    report = run(
        args.config.expanduser().resolve(),
        args.repo_id,
        args.norm_stats.expanduser().resolve(),
        args.model_path.expanduser().resolve(),
    )
    print(
        "OPENPI ACTUAL SFT BATCH PASSED "
        f"dataset={report['dataset']} "
        f"batch_actions={report['batch_action_shape']} "
        f"decoded_actions={report['decoded_action_shape']}"
    )
    print("BATCH AUDIT " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
