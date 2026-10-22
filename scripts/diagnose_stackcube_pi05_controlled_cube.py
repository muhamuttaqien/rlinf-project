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

"""Red-cube-only intervention: does Pi0.5 approach its actual object target?

Collect plausible red-cube XY targets from seeded StackCube resets, then
ALWAYS reset the original base scene with the same robot, camera and green
cube. Teleport ONLY red cube A to each target, preserving its z and rotation,
and re-render RGB with BaseEnv.get_obs() without advancing physics.

For each isolated position, predict the initial 10-action chunk using
fixed robot qpos and flow noise, then execute up to 120 policy-controlled
steps with fresh observations to measure real TCP approach, closing distance
and grasp. This is diagnostic only; no datasets, configs, model weights or
controller implementations are changed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts import evaluate_stackcube_pi05 as evaluator
from scripts.diagnose_stackcube_pi05_visual_sensitivity import (
    DEFAULT_CHECKPOINT,
    compare_actions,
    infer,
)
from scripts.diagnose_stackcube_pi05_object_generalization import xyz

DEFAULT_OUTPUT = Path(
    "/data1/Muttaqien/rlinf_data/results/"
    "maniskill_stackcube_panda_pi05_state_sft_resume100/"
    "evaluation/step_200/controlled_cube"
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5, 10000, 10001, 10002, 10003, 10004, 10005)


def select_targets(
    candidates: list[tuple[int, np.ndarray]],
    fixed_green: np.ndarray,
    fixed_tcp: np.ndarray,
    *,
    max_scenes: int,
    minimum_cube_separation_m: float = 0.075,
    minimum_tcp_distance_m: float = 0.085,
    minimum_candidate_xy_separation_m: float = 0.03,
) -> list[tuple[int, np.ndarray]]:
    """Only use distinct valid tabletop positions without initial collisions."""
    selected = []
    for seed, point in candidates:
        point = np.asarray(point, dtype=np.float64).reshape(3)
        if not np.isfinite(point).all():
            raise ValueError("Non-finite candidate cube position")
        if np.linalg.norm(point[:2] - fixed_green[:2]) < minimum_cube_separation_m:
            continue
        if np.linalg.norm(point - fixed_tcp) < minimum_tcp_distance_m:
            continue
        if any(
            np.linalg.norm(point[:2] - previous[:2])
            < minimum_candidate_xy_separation_m
            for _, previous in selected
        ):
            continue
        selected.append((int(seed), point.copy()))
        if len(selected) >= max_scenes:
            break
    return selected


def summarize(rollouts: list[dict[str, Any]], first_chunks: list[np.ndarray]) -> dict:
    if len(rollouts) < 4 or len(rollouts) != len(first_chunks):
        raise ValueError("At least four controlled rollouts/chunks are required")
    points = np.array([r["cubeA_initial_xyz_m"] for r in rollouts])
    from itertools import combinations

    all_pair_diffs = [
        compare_actions(first_chunks[i], first_chunks[j])
        for i, j in combinations(range(len(first_chunks)), 2)
    ]
    first = first_chunks[0]
    return {
        "scene_count": len(rollouts),
        "red_x_span_m": float(np.ptp(points[:, 0])),
        "red_y_span_m": float(np.ptp(points[:, 1])),
        "initial_arm_chunk_pairwise_mae_rad": float(np.mean([
            d["arm_mae_all_10_actions_rad"] for d in all_pair_diffs
        ])),
        "initial_gripper_chunk_pairwise_mae": float(np.mean([
            d["gripper_mae_all_10_actions"] for d in all_pair_diffs
        ])),
        "initial_arm_chunk_mae_vs_reference_rad": [
            compare_actions(first, actions)["arm_mae_all_10_actions_rad"]
            for actions in first_chunks
        ],
        "mean_tcp_distance_reduction_m": float(np.mean([
            r["initial_tcp_distance_m"] - r["final_tcp_distance_m"]
            for r in rollouts
        ])),
        "mean_best_tcp_distance_reduction_m": float(np.mean([
            r["initial_tcp_distance_m"] - r["min_tcp_distance_m"]
            for r in rollouts
        ])),
        "scenes_with_tcp_under_8cm": sum(
            r["min_tcp_distance_m"] < 0.08 for r in rollouts
        ),
        "scenes_with_grasp": sum(bool(r["ever_grasped"]) for r in rollouts),
        "early_closures_over_10cm": sum(
            r["first_closing_distance_m"] is not None
            and r["first_closing_distance_m"] > 0.10
            for r in rollouts
        ),
    }


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch
    from mani_skill.utils.structs.pose import Pose

    from rlinf.models.embodiment.openpi import get_model

    if len(args.seeds) < args.max_scenes or args.max_scenes < 4:
        raise ValueError("Provide at least max_scenes seeds and at least four scenes")
    if args.max_steps < 10 or args.max_steps > 300:
        raise ValueError("max_steps must be in [10,300]")
    ckpt = args.checkpoint.expanduser().resolve()
    stats = args.norm_stats.expanduser().resolve()
    if not (ckpt / "model_state_dict/full_weights.pt").is_file():
        raise FileNotFoundError(f"Missing checkpoint full weights: {ckpt}")
    if not stats.is_file():
        raise FileNotFoundError(f"Missing normalization stats: {stats}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Pi0.5 inference")
    cfg, _ = evaluator.make_model_config(
        args.config, args.model_config, ckpt, stats
    )
    model = get_model(cfg).to("cuda:0").eval()

    env = gym.make(
        "StackCube-v1", robot_uids="panda", num_envs=1, obs_mode="rgb",
        control_mode="pd_joint_pos", reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 200, "control_freq": 20},
        sensor_configs={"width": 224, "height": 224},
        max_episode_steps=300,
    )
    try:
        base = env.unwrapped
        env.reset(seed=args.base_seed)
        ref_green = xyz(base.cubeB.pose, label="reference green cube")
        ref_tcp = xyz(base.agent.tcp.pose, label="reference TCP")
        ref_red = xyz(base.cubeA.pose, label="reference red cube")
        ref_quat = base.cubeA.pose.q.detach().clone()
        ref_state = evaluator.extract_stackcube_obs(base.get_obs())[0]["states"][0].copy()

        candidates = [(args.base_seed, ref_red)]
        for seed in args.seeds:
            if seed == args.base_seed:
                continue
            env.reset(seed=int(seed))
            proposed = xyz(base.cubeA.pose, label=f"candidate red cube seed {seed}")
            proposed[2] = ref_red[2]
            candidates.append((seed, proposed))
        targets = select_targets(
            candidates, ref_green, ref_tcp, max_scenes=args.max_scenes
        )
        if len(targets) < 4:
            raise RuntimeError(
                f"Only {len(targets)} safe distinct cube positions from seed list; "
                "pass more --seeds"
            )

        rollouts = []
        initial_chunks = []
        baseline_image = None
        out_dir = args.output.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for seed, point in targets:
                env.reset(seed=args.base_seed)
                p = torch.as_tensor(
                    point[None, :], dtype=torch.float32, device=base.device
                )
                base.cubeA.set_pose(Pose.create_from_pq(p=p, q=ref_quat))
                obs = base.get_obs()  # refreshes RGB without any physics step
                model_obs, rgb = evaluator.extract_stackcube_obs(obs)
                state = model_obs["states"][0]
                if not np.allclose(state, ref_state, atol=1e-6):
                    raise RuntimeError("Robot qpos changed across controlled resets")
                if not np.allclose(
                    xyz(base.cubeB.pose, label="green cube"), ref_green, atol=1e-6
                ):
                    raise RuntimeError("Green cube moved across controlled resets")
                if not np.allclose(
                    xyz(base.agent.tcp.pose, label="TCP"), ref_tcp, atol=1e-6
                ):
                    raise RuntimeError("Robot TCP moved across controlled resets")
                if not np.allclose(
                    xyz(base.cubeA.pose, label="red cube"), point, atol=1e-6
                ):
                    raise RuntimeError("Red cube intervention was not applied")

                rgb = np.asarray(rgb, dtype=np.uint8).copy()
                if baseline_image is None:
                    baseline_image = rgb
                rgb_mae = float(np.abs(
                    rgb.astype(np.float32)
                    - baseline_image.astype(np.float32)
                ).mean())
                if args.save_images:
                    import imageio.v2 as imageio
                    imageio.imwrite(
                        str(out_dir / f"red_cube_seed_{seed}.png"), rgb
                    )
                initial_distance = float(np.linalg.norm(ref_tcp - point))
                min_distance = initial_distance
                current_distance = initial_distance
                first_closing_step = None
                first_closing_distance = None
                grasped = False
                first_actions = None
                steps = 0
                while steps < args.max_steps:
                    if first_actions is None:
                        actions = infer(
                            model, rgb, state, seed=args.sampling_seed
                        )
                        first_actions = actions.copy()
                    else:
                        model_obs, _ = evaluator.extract_stackcube_obs(obs)
                        actions = infer(
                            model, model_obs["main_images"][0],
                            model_obs["states"][0],
                            seed=args.sampling_seed + steps // 10,
                        )
                    for action in actions:
                        if steps >= args.max_steps:
                            break
                        obs, _reward, terminated, truncated, info = env.step(
                            action[None, :]
                        )
                        steps += 1
                        current_distance = float(np.linalg.norm(
                            xyz(base.agent.tcp.pose, label="current TCP")
                            - xyz(base.cubeA.pose, label="current red cube")
                        ))
                        min_distance = min(min_distance, current_distance)
                        grasped |= evaluator.scalar_bool(
                            info["is_cubeA_grasped"]
                        )
                        if first_closing_step is None and action[-1] < 0.0:
                            first_closing_step = steps
                            first_closing_distance = current_distance
                        if (
                            evaluator.scalar_bool(terminated)
                            or evaluator.scalar_bool(truncated)
                            or evaluator.success_from_info(info)
                        ):
                            break
                    if (
                        evaluator.scalar_bool(terminated)
                        or evaluator.scalar_bool(truncated)
                        or evaluator.success_from_info(info)
                    ):
                        break

                initial_chunks.append(first_actions)
                rollouts.append({
                    "candidate_seed": seed,
                    "cubeA_initial_xyz_m": point.tolist(),
                    "initial_tcp_distance_m": initial_distance,
                    "final_tcp_distance_m": current_distance,
                    "min_tcp_distance_m": min_distance,
                    "steps": steps,
                    "ever_grasped": bool(grasped),
                    "first_closing_step": first_closing_step,
                    "first_closing_distance_m": first_closing_distance,
                    "camera_mae_vs_baseline_0_to_255": rgb_mae,
                    "first_predicted_action_8d": first_actions[0].tolist(),
                    "last_initial_chunk_action_8d": first_actions[-1].tolist(),
                })

        stats_summary = summarize(rollouts, initial_chunks)
        if not any(r["camera_mae_vs_baseline_0_to_255"] > 0 for r in rollouts[1:]):
            raise RuntimeError(
                "Red cube moved but the camera images did not change; "
                "check ManiSkill sensor rendering"
            )
        report = {
            "experiment": "controlled_red_cube_only_with_short_closed_loop_rollout",
            "checkpoint": str(ckpt),
            "base_seed": args.base_seed,
            "source_seeds": list(args.seeds),
            "sampling_seed": args.sampling_seed,
            "fixed_robot_qpos_9d": ref_state.tolist(),
            "fixed_green_cube_xyz_m": ref_green.tolist(),
            "fixed_starting_tcp_xyz_m": ref_tcp.tolist(),
            "max_steps_per_scene": args.max_steps,
            "summary": stats_summary,
            "per_scene": rollouts,
        }
        saved = out_dir / "controlled_cube.json"
        saved.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            "CONTROLLED CUBE: "
            f"scenes={stats_summary['scene_count']} "
            f"red_x_span_m={stats_summary['red_x_span_m']:.3f} "
            f"red_y_span_m={stats_summary['red_y_span_m']:.3f} "
            "robot_and_green_fixed=yes"
        )
        print(
            "POLICY: "
            f"initial_arm_mae_rad={stats_summary['initial_arm_chunk_pairwise_mae_rad']:.5f} "
            f"initial_gripper_mae={stats_summary['initial_gripper_chunk_pairwise_mae']:.5f} "
            f"mean_best_approach_m={stats_summary['mean_best_tcp_distance_reduction_m']:.3f}"
        )
        print(
            "OUTCOME: "
            f"under_8cm={stats_summary['scenes_with_tcp_under_8cm']}/{len(rollouts)} "
            f"grasped={stats_summary['scenes_with_grasp']}/{len(rollouts)} "
            f"closing_above_10cm={stats_summary['early_closures_over_10cm']}/{len(rollouts)}"
        )
        print(f"Saved: {saved}")
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
    parser.add_argument("--model-config", type=Path, default=evaluator.MODEL_CONFIG)
    parser.add_argument(
        "--norm-stats", type=Path,
        default=Path(os.getenv(
            "STACKCUBE_NORM_STATS",
            "assets/pi05_stackcube_panda_joint/local/"
            "stackcube_sft_pilot100_lab/norm_stats.json"
        )),
    )
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-scenes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
