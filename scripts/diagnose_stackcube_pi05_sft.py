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

"""Teacher-forced StackCube Pi0.5 diagnostic on real saved expert frames.

No simulation rollouts, retraining, mutation of demonstration files, or changes
to the checkpoint. The two inference passes use identical initial flow noise.

IMPORTANT: the 2026-10-22 SFT checkpoint was trained using
preprocess_observation(train=True, rng=None), which applied a deterministic 95%
TOP-LEFT crop to the base camera. Standard inference uses train=False (no crop).
This script compares those two image distributions without changing the
production evaluator or postprocessing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path(
    "examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_easy_ee_delta_two_cam.yaml"
)
DEFAULT_MODEL_CONFIG = Path("examples/sft/config/model/pi0_5.yaml")
DEFAULT_ROOT = Path(
    "/data1/Muttaqien/rlinf_data/results/"
    "maniskill_stackcube_easy_ee_delta_two_cam_pi05_state_sft_rerun"
)
DEFAULT_STATS = Path(
    "/data1/Muttaqien/rlinf_data/pi05_stackcube_easy_ee_delta_two_cam_assets/"
    "local/stackcube_easy_ee_delta_two_cam100/norm_stats.json"
)


def action_metrics(expert: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    """Compare executable 7D controller actions without mixing arm/gripper."""
    expert = np.asarray(expert, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    if expert.shape != predicted.shape or expert.ndim != 2 or expert.shape[1] != 7:
        raise ValueError(
            f"Need matching [horizon,7] actions; got {expert.shape}, {predicted.shape}"
        )
    if not np.isfinite(expert).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite actions")
    return {
        "arm_mae": float(np.abs(expert[:, :6] - predicted[:, :6]).mean()),
        "gripper_mae": float(np.abs(expert[:, 6] - predicted[:, 6]).mean()),
        "gripper_close_agreement": float(
            np.mean((expert[:, 6] < 0) == (predicted[:, 6] < 0))
        ),
        "first_arm_mae": float(np.abs(expert[0, :6] - predicted[0, :6]).mean()),
        "first_gripper_command": float(predicted[0, 6]),
    }


def selected_indices(start: int, end: int, count: int) -> list[int]:
    """Sample phases evenly from one successful demonstration episode."""
    if end <= start or count < 1:
        raise ValueError("Expected non-empty episode and positive sample count")
    return np.unique(
        np.linspace(start, end - 1, min(count, end - start), dtype=np.int64)
    ).tolist()


def run(args: argparse.Namespace) -> None:
    import torch

    from rlinf.data.datasets.openpi.v3_stackcube_dataset import V3StackCubeDataset
    from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
    from scripts.evaluate_stackcube_pi05 import make_model_config
    from toolkits.lerobot.stackcube_ee_delta import policy_input

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required to load the trained Pi0.5 checkpoint")
    ckpt = args.checkpoint.expanduser().resolve()
    stats = args.norm_stats.expanduser().resolve()
    # Resolve exactly as the RLinf SFT loader does. The on-disk dataset
    # location can differ from HF_LEROBOT_HOME set in the current shell.
    root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else resolve_lerobot_dataset_root(args.repo_id)
    )
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"LeRobot dataset metadata not found under {root}. "
            "Pass --dataset-root with the existing training dataset's "
            "directory (the one containing meta/info.json); do not "
            "recollect or overwrite the dataset."
        )
    if not (ckpt / "model_state_dict/full_weights.pt").is_file():
        raise FileNotFoundError(f"Missing trained checkpoint at {ckpt}")
    if not stats.is_file():
        raise FileNotFoundError(f"Missing norm stats at {stats}")

    dataset = V3StackCubeDataset(
        root, action_horizon=10, use_wrist_image=True,
        state_dim=8, action_dim=7,
    )
    if args.episode not in dataset.episodes:
        raise ValueError(f"Episode {args.episode} not present in dataset")
    episode = dataset.episodes[args.episode]
    indices = selected_indices(
        int(episode["dataset_from_index"]),
        int(episode["dataset_to_index"]),
        args.samples,
    )

    cfg, _ = make_model_config(
        args.config, DEFAULT_MODEL_CONFIG, ckpt, stats,
    )
    model = get_model(cfg).to("cuda:0").eval()
    generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
    rows = []

    with torch.inference_mode():
        for index in indices:
            row = dataset[index]
            input_data = policy_input(row)
            observation = model.env_obs_to_observation(input_data)
            noise = torch.randn(
                1, model.action_horizon, model.action_dim,
                generator=generator, device=model.device,
            )

            # Normal inference (train=False) versus the ORIGINAL SFT
            # preprocessing (train=True, rng=None). Do not pass an RNG to
            # legacy_preprocessed: original SFT did not pass one either.
            normal = model.sample_actions(
                observation, num_steps=model.num_steps, noise=noise,
            )
            normal = model.decode_actions(normal, observation.state)
            legacy_preprocessed = preprocess_observation(
                observation, train=True,
            )
            legacy = model.sample_actions(
                legacy_preprocessed, num_steps=model.num_steps, noise=noise,
            )
            legacy = model.decode_actions(legacy, observation.state)

            target = row["actions"]
            normal_actions = normal[0].detach().cpu().numpy()
            legacy_actions = legacy[0].detach().cpu().numpy()
            result = {
                "dataset_index": index,
                "frame_index": int(row["frame_index"]),
                "expert_first_action": target[0].tolist(),
                "inference_first_action": normal_actions[0].tolist(),
                "training_view_first_action": legacy_actions[0].tolist(),
                "normal": action_metrics(target, normal_actions),
                "training_view": action_metrics(target, legacy_actions),
            }
            rows.append(result)
            print(
                f"FRAME {result['frame_index']:3d} "
                f"expert_grip={target[0, 6]:+.2f} "
                f"inference_grip={normal_actions[0, 6]:+.2f} "
                f"training_view_grip={legacy_actions[0, 6]:+.2f} "
                f"arm_mae_normal={result['normal']['arm_mae']:.3f} "
                f"arm_mae_training_view={result['training_view']['arm_mae']:.3f}"
            )

    averages = {}
    for view in ("normal", "training_view"):
        averages[view] = {
            key: float(np.mean([row[view][key] for row in rows]))
            for key in (
                "arm_mae", "gripper_mae", "gripper_close_agreement",
                "first_arm_mae",
            )
        }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(ckpt),
        "dataset_root": str(root),
        "episode": args.episode,
        "sample_seed": args.seed,
        "samples": len(rows),
        "mean": averages,
        "frames": rows,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(averages, sort_keys=True))
    print(f"SAVED {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_ROOT /
        "checkpoints/global_step_600/actor",
    )
    parser.add_argument(
        "--repo-id", default=os.getenv(
            "STACKCUBE_DATASET", "local/stackcube_easy_ee_delta_two_cam100",
        ),
        help="LeRobot repo ID passed to RLinf's canonical dataset resolver",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=None,
        help="Optional exact existing dataset directory (contains meta/info.json)",
    )
    parser.add_argument(
        "--norm-stats", type=Path,
        default=Path(os.getenv("STACKCUBE_NORM_STATS", str(DEFAULT_STATS))),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_ROOT /
        "evaluation/step_600/offline_expert_diagnostic.json",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
