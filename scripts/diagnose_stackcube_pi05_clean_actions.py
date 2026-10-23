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

"""Compare 7D policy first-action predictions on saved vs held-out expert frames.

Uses the REAL checkpoint, exact evaluation transforms, and original expert
commands. Does not modify datasets, run training, or execute policy actions.
This is teacher-forced inference, not closed-loop success estimation.

Sample a few frames across observable gripper phases. Labels are only
"before_close", "close_or_carry", and "after_reopen": the middle phase
does not distinguish grasping from lifting or transport.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.check_stackcube_pi05_clean_transforms import camera_to_hwc_uint8
from scripts.evaluate_stackcube_pi05 import as_numpy, make_model_config
from scripts.evaluate_stackcube_pi05_ee_delta import (
    CONFIG, MODEL_CONFIG, decode_first_action,
)
from toolkits.lerobot.stackcube_ee_delta import make_env, policy_input
from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root


def phase_samples(actions: np.ndarray) -> list[tuple[int, str]]:
    """Choose bounded, representative indices without assuming grasp timing."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
        raise ValueError(f"Expected [T,7] expert actions, got {actions.shape}")
    gripper = actions[:, 6]
    closes = np.flatnonzero(gripper < -0.5)
    if len(closes) == 0:
        raise ValueError("Expert trajectory has no gripper-close phase")
    first_close = int(closes[0])
    opens = np.flatnonzero(
        (np.arange(len(actions)) > first_close) & (gripper > 0.5)
    )
    if len(opens) == 0:
        raise ValueError("Expert trajectory has no release/reopen phase")
    first_reopen = int(opens[0])

    indices: list[tuple[int, str]] = []
    for label, start, stop, count in (
        ("before_close", 0, first_close, 3),
        ("close_or_carry", first_close, first_reopen, 4),
        ("after_reopen", first_reopen, len(actions), 2),
    ):
        if stop <= start:
            continue
        selected = np.linspace(
            start, stop - 1, num=min(count, stop - start), dtype=np.int64
        )
        indices.extend((int(index), label) for index in np.unique(selected))
    return indices


def _sample_record(sample: dict) -> dict:
    state = as_numpy(sample["state"]).astype(np.float32)
    action = as_numpy(sample["actions"]).astype(np.float32)
    if state.shape != (8,) or action.shape != (7,):
        raise ValueError(f"Expected 8D state / 7D action, got {state.shape} / {action.shape}")
    return {
        "image": camera_to_hwc_uint8(sample["image"]),
        "wrist_image": camera_to_hwc_uint8(sample["wrist_image"]),
        "state": state,
        "actions": action,
    }


def training_episode(repo_id: str, episode: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from rlinf.data.datasets.openpi.v3_stackcube_dataset import (
        is_lerobot_v3_dataset,
    )

    root = resolve_lerobot_dataset_root(repo_id)
    if is_lerobot_v3_dataset(root):
        raise ValueError("This diagnostic expects Clean V1 native LeRobot data")
    reader = LeRobotDataset(repo_id=repo_id, root=root)
    episode_ids = np.asarray(reader.hf_dataset["episode_index"], dtype=np.int64)
    indices = np.flatnonzero(episode_ids == episode)
    if not len(indices):
        raise ValueError(f"Episode {episode} not found in {repo_id}")
    expert_actions = np.asarray(
        reader.hf_dataset["actions"], dtype=np.float32
    )[indices]
    selected = phase_samples(expert_actions)
    return [
        (step, phase, _sample_record(reader[int(indices[step])]))
        for step, phase in selected
    ]


def heldout_episode(seed: int):
    from toolkits.lerobot.collect_stackcube_ee_delta import collect_episode

    env = make_env()
    try:
        frames = collect_episode(env, seed, max_steps=300)
    finally:
        env.close()
    if not frames:
        raise RuntimeError(
            f"Expert failed on held-out seed {seed}; do not compare to failed data"
        )
    expert_actions = np.asarray([f["actions"] for f in frames], dtype=np.float32)
    return [
        (step, phase, _sample_record(frames[step]))
        for step, phase in phase_samples(expert_actions)
    ]


def compare(model, cases, *, source: str, noise_seed: int, output: list[dict]):
    import torch

    for step, phase, frame in cases:
        rng = torch.Generator(device=model.device).manual_seed(noise_seed + step)
        predicted, _ = model.predict_action_batch(
            policy_input(frame), mode="eval", rng=rng
        )
        predicted = decode_first_action(predicted)
        target = frame["actions"]
        error = np.abs(predicted - target)
        record = {
            "source": source, "step": step, "phase": phase,
            "target": target.tolist(), "prediction": predicted.tolist(),
            "mae_7d": float(error.mean()),
            "mae_xyz": float(error[:3].mean()),
            "mae_rotation": float(error[3:6].mean()),
            "mae_gripper": float(error[6]),
            "abs_error_by_channel": error.tolist(),
        }
        output.append(record)
        print(
            f"{source:14s} step={step:3d} {phase:15s} "
            f"mae7={record['mae_7d']:.3f} "
            f"xyz={record['mae_xyz']:.3f} "
            f"rot={record['mae_rotation']:.3f} "
            f"grip={record['mae_gripper']:.3f} "
            f"target={np.round(target, 2).tolist()} "
            f"pred={np.round(predicted, 2).tolist()}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--norm-stats", type=Path, required=True)
    p.add_argument("--dataset", default="local/stackcube_pi05_clean_v1")
    p.add_argument("--train-episode", type=int, default=0)
    p.add_argument("--heldout-seed", type=int, default=10000)
    p.add_argument("--noise-seed", type=int, default=1234)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import torch
    from rlinf.models.embodiment.openpi import get_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for π₀.₅ action inference")
    if not (args.checkpoint / "model_state_dict" / "full_weights.pt").is_file():
        raise FileNotFoundError(f"Missing SFT checkpoint: {args.checkpoint}")
    if not args.norm_stats.is_file():
        raise FileNotFoundError(f"Missing normalization stats: {args.norm_stats}")

    train_cases = training_episode(args.dataset, args.train_episode)
    heldout_cases = heldout_episode(args.heldout_seed)
    cfg, _ = make_model_config(
        args.config, MODEL_CONFIG, args.checkpoint, args.norm_stats,
    )
    if cfg.openpi.config_name != "pi05_stackcube_panda_easy_ee_delta_two_cam_state":
        raise ValueError("Expected the Clean V1 7D EE-delta OpenPI configuration")
    model = get_model(cfg).to("cuda:0").eval()
    results: list[dict] = []
    with torch.inference_mode():
        compare(
            model, train_cases, source=f"train_ep{args.train_episode}",
            noise_seed=args.noise_seed, output=results,
        )
        compare(
            model, heldout_cases, source=f"heldout_{args.heldout_seed}",
            noise_seed=args.noise_seed + args.heldout_seed, output=results,
        )

    summaries = {}
    for source in sorted({row["source"] for row in results}):
        records = [row for row in results if row["source"] == source]
        summaries[source] = {
            "samples": len(records),
            "mae_7d": float(np.mean([r["mae_7d"] for r in records])),
            "mae_xyz": float(np.mean([r["mae_xyz"] for r in records])),
            "mae_rotation": float(np.mean([r["mae_rotation"] for r in records])),
            "mae_gripper": float(np.mean([r["mae_gripper"] for r in records])),
        }
        print(f"SUMMARY {source}: {json.dumps(summaries[source], sort_keys=True)}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summaries, "samples": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OFFLINE ACTION DIAGNOSTIC COMPLETE: {output}")


if __name__ == "__main__":
    main()
