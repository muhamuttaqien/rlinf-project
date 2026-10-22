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

"""Does the trained Pi0.5 policy change actions when only the RGB image changes?

Two ManiSkill scenes are reset with different seeds. Image A vs image B is
compared at *exactly the same* 9D robot state and task text, using identical
sampling noise. Repeat image A to measure stochastic/non-repeatable variation.
An additional same-image comparison swaps the state to check proprioceptive
sensitivity. This diagnostic does not execute any policy actions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts import evaluate_stackcube_pi05 as evaluator

DEFAULT_CHECKPOINT = Path(
    "/data1/Muttaqien/rlinf_data/results/"
    "maniskill_stackcube_panda_pi05_state_sft_resume100/"
    "checkpoints/global_step_200/actor"
)
DEFAULT_OUTPUT = Path(
    "/data1/Muttaqien/rlinf_data/results/"
    "maniskill_stackcube_panda_pi05_state_sft_resume100/"
    "evaluation/step_200/visual_sensitivity"
)


def make_input(image: Any, state: Any) -> dict[str, Any]:
    """Use the exact observation keys expected by RLinf's Pi0.5 adapter."""
    rgb = np.asarray(image, dtype=np.uint8)
    qpos = np.asarray(state, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {rgb.shape}")
    if qpos.shape != (9,) or not np.isfinite(qpos).all():
        raise ValueError(f"Expected finite 9D Panda state, got {qpos.shape}")
    return {
        "main_images": rgb[None].copy(),
        "states": qpos[None].copy(),
        "task_descriptions": [evaluator.TASK],
    }


def infer(model: Any, image: Any, state: Any, *, seed: int) -> np.ndarray:
    import torch

    # OpenPI Pi0.5 action generation samples noise; each comparison begins
    # with the same torch RNG state to isolate the input intervention.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    obs = make_input(image, state)
    predicted, _ = model.predict_action_batch(obs, mode="eval")
    actions = evaluator.actions_to_numpy(predicted, chunk=10)
    if actions.shape != (10, 8):
        raise ValueError(f"Expected 10x8 model actions, got {actions.shape}")
    return actions


def compare_actions(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Compare denormalized physical actions (radians + gripper command)."""
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.shape != (10, 8) or y.shape != (10, 8):
        raise ValueError("Expected matched 10x8 action chunks")
    return {
        "arm_mae_first_action_rad": float(np.abs(x[0, :7] - y[0, :7]).mean()),
        "arm_mae_all_10_actions_rad": float(np.abs(x[:, :7] - y[:, :7]).mean()),
        "gripper_abs_first_action": float(abs(x[0, 7] - y[0, 7])),
        "gripper_mae_all_10_actions": float(np.abs(x[:, 7] - y[:, 7]).mean()),
    }


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch

    from rlinf.models.embodiment.openpi import get_model

    ckpt = args.checkpoint.expanduser().resolve()
    stats = args.norm_stats.expanduser().resolve()
    if not (ckpt / "model_state_dict/full_weights.pt").is_file():
        raise FileNotFoundError(f"Missing model-only weights in {ckpt}")
    if not stats.is_file():
        raise FileNotFoundError(f"Missing norm stats: {stats}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU inference is required")

    model_cfg, _ = evaluator.make_model_config(
        args.config, args.model_config, ckpt, stats
    )
    model = get_model(model_cfg).to("cuda:0").eval()
    env = gym.make(
        "StackCube-v1", robot_uids="panda", num_envs=1,
        obs_mode="rgb", control_mode="pd_joint_pos", reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 200, "control_freq": 20},
        sensor_configs={"width": 224, "height": 224},
        max_episode_steps=300,
    )
    try:
        scene_a = evaluator.extract_stackcube_obs(env.reset(seed=args.seed_a)[0])[0]
        scene_b = evaluator.extract_stackcube_obs(env.reset(seed=args.seed_b)[0])[0]
        image_a = scene_a["main_images"][0]
        image_b = scene_b["main_images"][0]
        state_a = scene_a["states"][0]
        state_b = scene_b["states"][0]

        # The two image samples differ by both robot view and scene. Holding
        # state fixed in the model intervention isolates its image-input effect.
        image_mae = float(
            np.abs(image_a.astype(np.float32) - image_b.astype(np.float32)).mean()
        )
        if image_mae <= 0.0:
            raise ValueError("The two scene images are identical: cannot test sensitivity")

        with torch.inference_mode():
            baseline = infer(model, image_a, state_a, seed=args.noise_seed)
            repeated = infer(model, image_a, state_a, seed=args.noise_seed)
            image_swapped = infer(model, image_b, state_a, seed=args.noise_seed)
            state_swapped = infer(model, image_a, state_b, seed=args.noise_seed)

        repeat = compare_actions(baseline, repeated)
        image = compare_actions(baseline, image_swapped)
        state = compare_actions(baseline, state_swapped)
        summary = {
            "checkpoint": str(ckpt),
            "seed_a": int(args.seed_a),
            "seed_b": int(args.seed_b),
            "sampling_seed": int(args.noise_seed),
            "camera_rgb_mae_0_to_255": image_mae,
            "robot_state_l2_rad": float(np.linalg.norm(state_a - state_b)),
            "repeat_same_observation": repeat,
            "swap_image_keep_state": image,
            "swap_state_keep_image": state,
            "baseline_first_action": baseline[0].tolist(),
            "image_swap_first_action": image_swapped[0].tolist(),
            "state_swap_first_action": state_swapped[0].tolist(),
        }
        out = args.output.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "visual_sensitivity.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "VISUAL SENSITIVITY: "
            f"image_rgb_mae={image_mae:.2f}/255 "
            f"repeat_arm_mae={repeat['arm_mae_all_10_actions_rad']:.5f} "
            f"image_swap_arm_mae={image['arm_mae_all_10_actions_rad']:.5f} "
            f"state_swap_arm_mae={state['arm_mae_all_10_actions_rad']:.5f}"
        )
        print(
            "GRIPPER: "
            f"repeat_mae={repeat['gripper_mae_all_10_actions']:.5f} "
            f"image_swap_mae={image['gripper_mae_all_10_actions']:.5f} "
            f"state_swap_mae={state['gripper_mae_all_10_actions']:.5f}"
        )
        print(f"Saved: {out / 'visual_sensitivity.json'}")
        if (
            repeat["arm_mae_all_10_actions_rad"] > 1e-4
            or repeat["gripper_mae_all_10_actions"] > 1e-4
        ):
            print(
                "CAUTION: same-observation repeat is not deterministic "
                "despite seeded sampling; interpret swap results carefully"
            )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config", type=Path,
        default=Path(
            "examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_state.yaml"
        ),
    )
    parser.add_argument(
        "--model-config", type=Path, default=evaluator.MODEL_CONFIG
    )
    parser.add_argument(
        "--norm-stats", type=Path,
        default=Path(os.environ.get(
            "STACKCUBE_NORM_STATS",
            "assets/pi05_stackcube_panda_joint/local/"
            "stackcube_sft_pilot100_lab/norm_stats.json",
        )),
    )
    parser.add_argument("--seed-a", type=int, default=0)
    parser.add_argument("--seed-b", type=int, default=10000)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
