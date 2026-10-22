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

"""StackCube action/gripper diagnostics for the existing pi05 SFT checkpoint.

Offline: predict on actual saved expert observations and compare the first
decoded joint-position target with the demonstration's corresponding action.
Online: record each predicted pd_joint_pos target and the simulator's actual
9D Panda qpos, including both finger joints. Never change controllers or data.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts import evaluate_stackcube_pi05 as evaluator

LOGGER = logging.getLogger(__name__)


def qpos_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Read 7 arm + 2 finger joint positions without decoding camera frames."""
    qpos = evaluator.as_numpy(obs["agent"]["qpos"]).reshape(-1)
    if qpos.shape != (9,) or not np.isfinite(qpos).all():
        raise ValueError(f"Expected finite 9D Panda qpos, got {qpos.shape}")
    return np.asarray(qpos, dtype=np.float32)


def trace_action(
    *,
    seed: int,
    step: int,
    target: Any,
    before: Any,
    after: Any,
    success: bool,
) -> dict[str, Any]:
    """Keep commanded gripper and measured finger qpos in separate units."""
    action = np.asarray(target, dtype=np.float32).reshape(-1)
    qbefore = np.asarray(before, dtype=np.float32).reshape(-1)
    qafter = np.asarray(after, dtype=np.float32).reshape(-1)
    if (
        action.shape != (8,)
        or qbefore.shape != (9,)
        or qafter.shape != (9,)
        or not np.isfinite(action).all()
        or not np.isfinite(qbefore).all()
        or not np.isfinite(qafter).all()
    ):
        raise ValueError("Invalid or non-finite 8D action / 9D Panda qpos")

    return {
        "seed": int(seed),
        "step": int(step),
        "success": bool(success),
        "arm_target": action[:7].tolist(),
        "arm_qpos_before": qbefore[:7].tolist(),
        "arm_qpos_after": qafter[:7].tolist(),
        "arm_target_mae_before": float(np.mean(np.abs(action[:7] - qbefore[:7]))),
        "arm_target_mae_after": float(np.mean(np.abs(action[:7] - qafter[:7]))),
        "arm_motion_l2": float(np.linalg.norm(qafter[:7] - qbefore[:7])),
        "gripper_command": float(action[7]),
        "finger_qpos_before": qbefore[7:].tolist(),
        "finger_qpos_after": qafter[7:].tolist(),
        "finger_qpos_sum": float(np.sum(qafter[7:])),
    }


class TracedEnv:
    """Forward ManiSkill calls unchanged and trace actions actually executed."""

    def __init__(self, env: Any):
        self.env = env
        self.seed: int | None = None
        self.obs: dict[str, Any] | None = None
        self.rows: list[dict[str, Any]] = []

    def reset(self, *, seed: int):
        result = self.env.reset(seed=seed)
        self.seed = seed
        self.obs = result[0]
        self.rows = []
        return result

    def step(self, action: Any):
        if self.obs is None or self.seed is None:
            raise RuntimeError("Reset TracedEnv before calling step")
        before = qpos_from_obs(self.obs)
        result = self.env.step(action)
        new_obs, _reward, _terminated, _truncated, info = result
        after = qpos_from_obs(new_obs)
        action_vector = evaluator.as_numpy(action).reshape(-1)
        self.rows.append(
            trace_action(
                seed=self.seed,
                step=len(self.rows) + 1,
                target=action_vector,
                before=before,
                after=after,
                success=evaluator.success_from_info(info),
            )
        )
        self.obs = new_obs
        return result


def expert_sample_indices(
    episode_indices: Any, *, episodes: int = 2, per_episode: int = 3
) -> list[tuple[int, int]]:
    """Sample deterministic, episode-local expert observations."""
    values = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    if episodes < 1 or per_episode < 1:
        raise ValueError("episodes and per_episode must be positive")
    result = []
    for ep in np.unique(values)[:episodes]:
        available = np.flatnonzero(values == ep)
        positions = np.linspace(
            0, len(available) - 1, min(per_episode, len(available)), dtype=int
        )
        result.extend((int(ep), int(available[p])) for p in positions)
    return result


def _read_expert_dataset(repo_id: str):
    """Use the same local v3 reader as the OpenPI SFT loader, or LeRobot v2."""
    from rlinf.data.datasets.openpi.v3_stackcube_dataset import (
        V3StackCubeDataset,
        is_lerobot_v3_dataset,
    )
    from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root

    root = resolve_lerobot_dataset_root(repo_id)
    if is_lerobot_v3_dataset(root):
        dataset = V3StackCubeDataset(root, action_horizon=10)
        episode_ids = [int(row["episode_index"]) for row in dataset.frames]
    else:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id, root=root)
        if "episode_index" not in dataset.hf_dataset.column_names:
            raise ValueError("Expert dataset has no episode_index column")
        episode_ids = dataset.hf_dataset["episode_index"]
    return dataset, episode_ids


def _to_expert_action(value: Any) -> np.ndarray:
    """Choose the first target, aligned to this saved expert observation."""
    raw = evaluator.as_numpy(value)
    if raw.shape == (8,):
        action = raw
    elif raw.ndim == 2 and raw.shape[1] == 8 and raw.shape[0] >= 1:
        action = raw[0]
    else:
        raise ValueError(f"Expected expert action [8] or [T,8], got {raw.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Non-finite expert action")
    return np.asarray(action, dtype=np.float32)


def compare_expert_predictions(
    model: Any, repo_id: str, *, max_episodes: int, samples_per_episode: int
) -> list[dict[str, Any]]:
    """Paired evaluation at *identical* observations, not distribution matching."""
    from rlinf.utils.stackcube_wandb_media import to_rgb_uint8

    dataset, episodes = _read_expert_dataset(repo_id)
    result = []
    for episode, idx in expert_sample_indices(
        episodes, episodes=max_episodes, per_episode=samples_per_episode
    ):
        sample = dataset[idx]
        state = np.asarray(evaluator.as_numpy(sample["state"]), dtype=np.float32)
        image = to_rgb_uint8(sample["image"])
        if state.shape != (9,) or image.shape[-1] != 3:
            raise ValueError(f"Invalid expert observation at sample {idx}")

        obs = {
            "main_images": image[None, ...],
            "states": state[None, ...],
            "task_descriptions": [evaluator.TASK],
        }
        predicted, _ = model.predict_action_batch(obs, mode="eval")
        action = evaluator.actions_to_numpy(predicted, chunk=1)[0]
        expert = _to_expert_action(sample["actions"])

        result.append(
            {
                "episode": episode,
                "dataset_index": idx,
                "expert_action": expert.tolist(),
                "predicted_action": action.tolist(),
                "arm_action_mae": float(np.mean(np.abs(action[:7] - expert[:7]))),
                "gripper_action_abs_error": float(abs(action[7] - expert[7])),
                "expert_gripper_command": float(expert[7]),
                "predicted_gripper_command": float(action[7]),
            }
        )
    return result


def summarize_diagnostics(
    expert_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    def average(rows, key):
        return float(np.mean([row[key] for row in rows])) if rows else None

    return {
        "paired_expert_samples": len(expert_rows),
        "paired_arm_action_mae": average(expert_rows, "arm_action_mae"),
        "paired_gripper_action_abs_error": average(
            expert_rows, "gripper_action_abs_error"
        ),
        "rollout_action_steps": len(rollout_rows),
        "mean_arm_target_mae_after": average(
            rollout_rows, "arm_target_mae_after"
        ),
        "mean_arm_motion_l2": average(rollout_rows, "arm_motion_l2"),
        "mean_gripper_command": average(rollout_rows, "gripper_command"),
        "mean_finger_qpos_sum": average(rollout_rows, "finger_qpos_sum"),
    }


def save_action_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Keep the trace readable without dumping thousands of terminal lines."""
    columns = (
        "seed", "step", "success", "gripper_command", "finger_qpos_sum",
        "arm_target_mae_before", "arm_target_mae_after", "arm_motion_l2",
        "arm_target", "arm_qpos_before", "arm_qpos_after",
        "finger_qpos_before", "finger_qpos_after",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def log_wandb_diagnostics(run, wandb, expert_rows, rollout_rows, summary):
    """Upload bounded paired-action tables and action/finger trajectories."""
    data = {
        f"diagnostics/{key}": value
        for key, value in summary.items()
        if value is not None
    }

    if expert_rows:
        table = wandb.Table(
            columns=[
                "expert_episode", "dataset_index", "expert_gripper",
                "predicted_gripper", "arm_mae", "gripper_abs_error",
            ],
            data=[
                [
                    row["episode"], row["dataset_index"],
                    row["expert_gripper_command"],
                    row["predicted_gripper_command"],
                    row["arm_action_mae"],
                    row["gripper_action_abs_error"],
                ]
                for row in expert_rows
            ],
        )
        data["diagnostics/paired_expert_actions"] = table
        data["diagnostics/paired_gripper_scatter"] = wandb.plot.scatter(
            table, "expert_gripper", "predicted_gripper",
            title="Same-observation expert vs policy gripper targets",
        )

    if rollout_rows:
        first_seed = rollout_rows[0]["seed"]
        rows = [r for r in rollout_rows if r["seed"] == first_seed]
        x = [r["step"] for r in rows]
        data["diagnostics/gripper_command_trace"] = wandb.plot.line_series(
            xs=x,
            ys=[[r["gripper_command"] for r in rows]],
            keys=["gripper command (controller target)"],
            title=f"Rollout seed {first_seed}: commanded gripper",
            xname="Simulator step",
        )
        data["diagnostics/finger_qpos_trace"] = wandb.plot.line_series(
            xs=x,
            ys=[
                [r["finger_qpos_after"][0] for r in rows],
                [r["finger_qpos_after"][1] for r in rows],
            ],
            keys=["finger joint 1 qpos", "finger joint 2 qpos"],
            title=f"Rollout seed {first_seed}: measured finger joints",
            xname="Simulator step",
        )
        data["diagnostics/arm_target_tracking_trace"] = wandb.plot.line_series(
            xs=x,
            ys=[[r["arm_target_mae_after"] for r in rows]],
            keys=["mean absolute joint target tracking error"],
            title=f"Rollout seed {first_seed}: arm target tracking",
            xname="Simulator step",
        )
    run.log(data, step=0)


def run(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    norm_stats = args.norm_stats.expanduser().resolve()
    weights = (
        checkpoint / "model.safetensors",
        checkpoint / "model_state_dict" / "full_weights.pt",
        checkpoint / "actor" / "model_state_dict" / "full_weights.pt",
    )
    if not any(weight.is_file() for weight in weights):
        raise FileNotFoundError(f"Cannot locate checkpoint weights: {checkpoint}")
    if not norm_stats.is_file():
        raise FileNotFoundError(f"Missing normalization stats: {norm_stats}")
    if min(args.episodes, args.max_steps, args.expert_episodes, args.expert_samples) < 1:
        raise ValueError("Episode, step, and expert sample counts must be positive")

    from rlinf.models.embodiment.openpi import get_model
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch

    cfg, run_cfg = evaluator.make_model_config(
        args.config, args.model_config, checkpoint, norm_stats
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; retain working SFT GPU environment")

    model = get_model(cfg).to("cuda:0").eval()
    env = gym.make(
        "StackCube-v1",
        robot_uids="panda", num_envs=1, obs_mode="rgb",
        control_mode="pd_joint_pos", reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 200, "control_freq": 20},
        sensor_configs={"width": 224, "height": 224},
        max_episode_steps=args.max_steps,
    )
    traced = TracedEnv(env)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo_id = os.environ.get("STACKCUBE_DATASET", "local/stackcube_sft_pilot100_lab")
    wb_run = None

    try:
        if not args.no_wandb:
            import wandb

            wb_run = wandb.init(
                project=str(run_cfg.runner.logger.project_name),
                name=f"maniskill_stackcube_panda_pi05_{args.label}_diagnostics",
                dir=str(output),
                config={
                    "checkpoint": str(checkpoint),
                    "norm_stats": str(norm_stats),
                    "train_dataset": repo_id,
                    "eval_seed_start": args.seed_start,
                    "episodes": args.episodes,
                    "expert_episodes": args.expert_episodes,
                    "expert_samples_per_episode": args.expert_samples,
                },
            )

        all_traces = []
        metrics = []
        with torch.inference_mode():
            expert_rows = compare_expert_predictions(
                model, repo_id,
                max_episodes=args.expert_episodes,
                samples_per_episode=args.expert_samples,
            )
            for offset in range(args.episodes):
                seed = args.seed_start + offset
                torch.manual_seed(seed)  # Reproducible model noise on each seed.
                result, frames = evaluator.evaluate_episode(
                    traced, model, seed, max_steps=args.max_steps,
                    record_video=offset == 0, video_stride=2,
                )
                metrics.append(result)
                all_traces.extend(traced.rows)
                LOGGER.info(
                    "seed=%d success=%s steps=%d target-MAE=%.4f",
                    seed, result["success"], result["steps"],
                    float(np.mean([
                        row["arm_target_mae_after"] for row in traced.rows
                    ])) if traced.rows else float("nan"),
                )
                if frames:
                    video_path = output / f"policy_seed_{seed}.mp4"
                    evaluator.save_video(frames, video_path, fps=10)
                    if wb_run is not None:
                        wb_run.log({
                            "diagnostics/policy_rollout_video":
                                wandb.Video(str(video_path), format="mp4")
                        }, step=0)

        summary = summarize_diagnostics(expert_rows, all_traces)
        summary["rollout_episodes"] = args.episodes
        summary["rollout_successes"] = sum(r["success"] for r in metrics)
        output_data = {
            "summary": summary,
            "episodes": metrics,
            "expert_pairs": expert_rows,
            "rollout_actions": all_traces,
        }
        (output / "diagnostics.json").write_text(
            json.dumps(output_data, indent=2) + "\n", encoding="utf-8"
        )
        save_action_csv(all_traces, output / "rollout_actions.csv")
        if wb_run is not None:
            try:
                log_wandb_diagnostics(
                    wb_run, wandb, expert_rows, all_traces, summary
                )
            except Exception as exc:
                LOGGER.warning("W&B diagnostic charts unavailable: %s", exc)
            wb_run.summary.update({
                f"diagnostics/{key}": value for key, value in summary.items()
                if value is not None
            })
        print(
            f"DIAGNOSTICS: paired_arm_mae={summary['paired_arm_action_mae']}, "
            f"paired_gripper_error={summary['paired_gripper_action_abs_error']}, "
            f"rollout_successes={summary['rollout_successes']}/{args.episodes}"
        )
        print(f"Saved: {output / 'diagnostics.json'}")
        print(f"Saved: {output / 'rollout_actions.csv'}")
    finally:
        env.close()
        if wb_run is not None:
            wb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=evaluator.CHECKPOINT)
    parser.add_argument(
        "--norm-stats", type=Path,
        default=Path(os.environ.get(
            "STACKCUBE_NORM_STATS",
            "assets/pi05_stackcube_panda_joint/local/"
            "stackcube_sft_pilot100_lab/norm_stats.json",
        )),
    )
    parser.add_argument("--config", type=Path, default=evaluator.CONFIG)
    parser.add_argument("--model-config", type=Path, default=evaluator.MODEL_CONFIG)
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "../results/maniskill_stackcube_panda_pi05_sft/"
            "evaluation/step_200/diagnostics"
        ),
    )
    parser.add_argument("--label", default="step200")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--expert-episodes", type=int, default=2)
    parser.add_argument("--expert-samples", type=int, default=3)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
