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

"""Evaluate StackCube OpenPI Pi0.5 SFT using *policy-generated* ManiSkill rollouts.

Run from the repository root with the same CUDA/NVML environment used for SFT.
The baseline can be evaluated by setting --checkpoint to PI05_MODEL_PATH.
This script neither trains a model nor replaces the original demonstration data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root

LOGGER = logging.getLogger(__name__)
TASK = "stack the red cube on top of the green cube and release it"
CHECKPOINT = Path(
    "../results/maniskill_stackcube_panda_pi05_sft/"
    "checkpoints/global_step_200/actor"
)
CONFIG = Path("examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05.yaml")
MODEL_CONFIG = Path("examples/sft/config/model/pi0_5.yaml")


def as_numpy(value: Any) -> np.ndarray:
    """Safely read a ManiSkill tensor or ndarray without its autograd context."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def success_from_info(info: Any) -> bool:
    """Read the simulator success flag, never use reward as a proxy."""
    if not isinstance(info, dict) or "success" not in info:
        raise KeyError("ManiSkill info is missing the 'success' flag")
    return bool(as_numpy(info["success"]).reshape(-1)[0])


def scalar_bool(value: Any) -> bool:
    return bool(as_numpy(value).reshape(-1)[0])


def actions_to_numpy(actions: Any, *, chunk: int = 10) -> np.ndarray:
    """Validate one batch of absolute Panda pd_joint_pos action targets."""
    arr = as_numpy(actions)
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[2] != 8:
        raise ValueError(f"Expected policy actions [1, horizon, 8], got {arr.shape}")
    if arr.shape[1] < 1 or not np.isfinite(arr).all():
        raise ValueError("Policy generated empty or non-finite actions")
    return np.asarray(arr[0, :chunk], dtype=np.float32)


def make_model_config(
    config_path: Path,
    model_config_path: Path,
    checkpoint: Path,
    norm_stats: Path,
):
    """Use the same YAML/model architecture as SFT, with task=eval."""
    from omegaconf import OmegaConf

    config = OmegaConf.merge(
        {"actor": {"model": OmegaConf.load(model_config_path)}},
        OmegaConf.load(config_path),
    )
    model_cfg = config.actor.model
    model_cfg.model_path = str(checkpoint.resolve())
    model_cfg.openpi.task = "eval"
    model_cfg.openpi.torch_compile = False
    model_cfg.openpi_data.norm_stats_path = str(norm_stats.resolve())
    if str(model_cfg.openpi.config_name) != "pi05_stackcube_panda_joint":
        raise ValueError("This evaluator requires pi05_stackcube_panda_joint")
    if int(model_cfg.action_dim) != 8 or int(model_cfg.openpi.action_horizon) != 10:
        raise ValueError("Expected 8D absolute actions and horizon 10")
    if int(model_cfg.openpi.num_images_in_input) != 1:
        raise ValueError("Expected base-camera-only StackCube training")
    OmegaConf.resolve(model_cfg)
    return model_cfg, config


def extract_stackcube_obs(obs: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Match the LeRobot StackCube collector's 9D state and camera selection."""
    from toolkits.lerobot.collect_maniskill_stackcube_recap import (
        _extract_observation,
    )

    parsed = _extract_observation(obs)
    image = parsed["image"]
    state = parsed["state"]
    if image.ndim != 3 or image.shape[-1] != 3 or state.shape != (9,):
        raise ValueError("Invalid StackCube base camera or 9D Panda state")
    env_obs = {
        "main_images": image[None, ...],
        "states": state[None, ...],
        "task_descriptions": [TASK],
    }
    return env_obs, image


def evaluate_episode(
    env: Any,
    model: Any,
    seed: int,
    *,
    max_steps: int,
    record_video: bool,
    video_stride: int = 2,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Roll out one held-out seed; compute success exclusively from ManiSkill."""
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    success = False
    steps = 0
    finished = False

    while steps < max_steps and not finished:
        model_obs, camera = extract_stackcube_obs(obs)
        if record_video and (not frames):
            frames.append(camera.copy())

        predicted, _ = model.predict_action_batch(model_obs, mode="eval")
        action_chunk = actions_to_numpy(predicted)
        for action in action_chunk:
            if steps >= max_steps:
                break
            # The simulator was created with num_envs=1. Supply one 8D target.
            obs, _reward, terminated, truncated, info = env.step(action[None, :])
            steps += 1
            success |= success_from_info(info)
            finished = scalar_bool(terminated) or scalar_bool(truncated)
            if record_video and (steps % video_stride == 0 or finished):
                _, camera = extract_stackcube_obs(obs)
                frames.append(camera.copy())
            if finished or success:
                break
        if success:
            break

    return {"seed": seed, "success": success, "steps": steps}, frames


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Write a playable local MP4 using the installed imageio-ffmpeg backend."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def run(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    stats = args.norm_stats.expanduser().resolve()
    if not stats.is_file():
        raise FileNotFoundError(f"Missing training normalization stats: {stats}")
    if not (
        (checkpoint / "model.safetensors").is_file()
        or (checkpoint / "model_state_dict" / "full_weights.pt").is_file()
        or (checkpoint / "actor" / "model_state_dict" / "full_weights.pt").is_file()
    ):
        raise FileNotFoundError(f"No OpenPI model weights at {checkpoint}")

    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--episodes and --max-steps must be positive")
    if args.max_videos < 0 or args.video_stride <= 0:
        raise ValueError("--max-videos must be nonnegative and --video-stride positive")

    model_cfg, config = make_model_config(
        args.config, args.model_config, checkpoint, stats
    )
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Validate the local training dataset exists; the evaluator does not reuse
    # any of its observations or the expert solver in the evaluation rollouts.
    from rlinf.data.storage.lerobot import resolve_lerobot_repo_id

    train_repo = resolve_lerobot_repo_id(os.getenv("STACKCUBE_DATASET", "local/stackcube_sft_pilot100_lab"))
    dataset_root = resolve_lerobot_dataset_root(train_repo)
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Training dataset not found at {dataset_root}")

    import torch
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    from rlinf.models.embodiment.openpi import get_model

    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference requires CUDA; check CUDA_VISIBLE_DEVICES")

    LOGGER.info("Loading %s onto cuda:0", checkpoint)
    model = get_model(model_cfg).to("cuda:0").eval()
    env = gym.make(
        "StackCube-v1",
        robot_uids="panda",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 200, "control_freq": 20},
        sensor_configs={"width": 224, "height": 224},
        max_episode_steps=args.max_steps,
    )

    run_name = f"maniskill_stackcube_panda_pi05_{args.label}_eval"
    wandb_run = None
    results = []
    try:
        if not args.no_wandb:
            import wandb

            wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=str(config.runner.logger.project_name),
                name=run_name,
                dir=str(out),
                config={
                    "model_checkpoint": str(checkpoint),
                    "normalization_stats": str(stats),
                    "dataset": train_repo,
                    "eval_episodes": args.episodes,
                    "seed_start": args.seed_start,
                    "max_steps": args.max_steps,
                    "action_dim": 8,
                    "action_chunk": 10,
                    "obs_image_size": 224,
                    "video_stride": args.video_stride,
                    "policy_type": "openpi_pi05",
                },
            )

        with torch.inference_mode():
            for episode_idx in range(args.episodes):
                seed = args.seed_start + episode_idx
                record = episode_idx < args.max_videos
                metrics, frames = evaluate_episode(
                    env,
                    model,
                    seed,
                    max_steps=args.max_steps,
                    record_video=record,
                    video_stride=args.video_stride,
                )
                results.append(metrics)
                LOGGER.info(
                    "Evaluation %d/%d seed=%d success=%s steps=%d",
                    episode_idx + 1, args.episodes, seed,
                    metrics["success"], metrics["steps"],
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "eval/episode_success": int(metrics["success"]),
                            "eval/episode_steps": metrics["steps"],
                            "eval/running_success_rate": float(
                                sum(x["success"] for x in results) / len(results)
                            ),
                        },
                        step=episode_idx,
                    )
                if frames:
                    video_file = out / "videos" / f"seed_{seed:05d}.mp4"
                    try:
                        save_video(frames, video_file, fps=max(1, 20 // args.video_stride))
                        if wandb_run is not None:
                            import wandb

                            wandb_run.log(
                                {
                                    f"eval/policy_rollout_seed_{seed}": wandb.Video(
                                        str(video_file), format="mp4",
                                        caption=(
                                            f"POLICY rollout seed={seed}; "
                                            f"success={metrics['success']}"
                                        ),
                                    )
                                },
                                step=episode_idx,
                            )
                    except Exception as exc:
                        LOGGER.warning("Could not save/upload video %s: %s", seed, exc)

        successes = sum(x["success"] for x in results)
        summary = {
            "label": args.label,
            "checkpoint": str(checkpoint),
            "episodes": len(results),
            "successes": successes,
            "success_rate": successes / len(results),
            "mean_episode_steps": float(np.mean([x["steps"] for x in results])),
            "seeds": [r["seed"] for r in results],
            "per_episode": results,
        }
        (out / "results.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "eval/success_rate": summary["success_rate"],
                    "eval/successes": successes,
                    "eval/episodes": len(results),
                    "eval/mean_episode_steps": summary["mean_episode_steps"],
                }
            )
        print(
            f"RESULT: {successes}/{len(results)} successful "
            f"({summary['success_rate']:.1%}); saved {out / 'results.json'}"
        )
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--norm-stats", type=Path, default=Path(
        os.getenv("STACKCUBE_NORM_STATS", "assets/pi05_stackcube_panda_joint/local/"
                  "stackcube_sft_pilot100_lab/norm_stats.json")))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIG)
    parser.add_argument("--output", type=Path, default=Path(
        "../results/maniskill_stackcube_panda_pi05_sft/evaluation/step_200"))
    parser.add_argument("--label", default="step200")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-videos", type=int, default=3)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
