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

"""Probe whether StackCube Pi0.5 actions vary with cube position across scenes.

Reset ManiSkill at distinct deterministic seeds, record real red/green cube and
TCP world coordinates and base camera pixels, and predict a 10-step action
chunk for each image while ALWAYS passing the first scene's 9D robot qpos
to the model. The prompt and flow sampling noise are also fixed.

This is a *sensitivity* diagnostic, not a proof of goal-directedness:
reinitializing a scene can change more than the cube position. In particular
we report image differences and the variation in actual robot initial qpos
because the rendered robot could still change even though the model receives
fixed proprioception. No model actions are executed and no poses are modified.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from scripts import evaluate_stackcube_pi05 as evaluator
from scripts.diagnose_stackcube_pi05_visual_sensitivity import (
    DEFAULT_CHECKPOINT,
    compare_actions,
    infer,
)

DEFAULT_OUTPUT = Path(
    "/data1/Muttaqien/rlinf_data/results/"
    "maniskill_stackcube_panda_pi05_state_sft_resume100/"
    "evaluation/step_200/object_generalization"
)
DEFAULT_SEEDS = (0, 1, 2, 3, 10000, 10001, 10002, 10003)


def xyz(pose: Any, *, label: str) -> np.ndarray:
    """Extract one ManiSkill environment's 3D world coordinate."""
    arr = evaluator.as_numpy(pose.p)
    if arr.size != 3:
        raise ValueError(f"{label}: expected a single 3D pose, got {arr.shape}")
    coordinate = np.asarray(arr, dtype=np.float64).reshape(3)
    if not np.isfinite(coordinate).all():
        raise ValueError(f"{label}: non-finite coordinate")
    return coordinate.copy()


def capture_scene(env: Any, seed: int) -> dict[str, Any]:
    """Read RGB/state from the SAME ManiSkill observation and actor poses."""
    observation, _info = env.reset(seed=int(seed))
    model_obs, _camera = evaluator.extract_stackcube_obs(observation)
    base = env.unwrapped
    return {
        "seed": int(seed),
        "image": np.asarray(model_obs["main_images"][0], dtype=np.uint8).copy(),
        "actual_qpos": np.asarray(model_obs["states"][0], dtype=np.float32).copy(),
        "tcp_xyz_m": xyz(base.agent.tcp.pose, label="tcp"),
        "cubeA_xyz_m": xyz(base.cubeA.pose, label="cubeA"),
        "cubeB_xyz_m": xyz(base.cubeB.pose, label="cubeB"),
    }


def pairwise_metrics(
    snapshots: list[dict[str, Any]],
    predictions: list[np.ndarray],
) -> list[dict[str, Any]]:
    """Compare scene distances with changes to the actions selected at fixed qpos."""
    if len(snapshots) != len(predictions) or len(snapshots) < 2:
        raise ValueError("Need at least two matching scenes/predictions")
    pairs = []
    for i, j in combinations(range(len(snapshots)), 2):
        a, b = snapshots[i], snapshots[j]
        action = compare_actions(predictions[i], predictions[j])
        pairs.append({
            "seed_a": a["seed"],
            "seed_b": b["seed"],
            "cubeA_xy_separation_m": float(
                np.linalg.norm(a["cubeA_xyz_m"][:2] - b["cubeA_xyz_m"][:2])
            ),
            "cubeB_xy_separation_m": float(
                np.linalg.norm(a["cubeB_xyz_m"][:2] - b["cubeB_xyz_m"][:2])
            ),
            "actual_qpos_separation_l2": float(
                np.linalg.norm(a["actual_qpos"] - b["actual_qpos"])
            ),
            "camera_mae_0_to_255": float(np.abs(
                a["image"].astype(np.float32)
                - b["image"].astype(np.float32)
            ).mean()),
            **action,
        })
    return pairs


def descriptive_pairwise_corr(pairs: list[dict[str, Any]]) -> float | None:
    """Descriptive only: pairwise distances are statistically dependent."""
    if len(pairs) < 3:
        return None
    x = np.array([p["cubeA_xy_separation_m"] for p in pairs])
    y = np.array([p["arm_mae_all_10_actions_rad"] for p in pairs])
    if float(np.std(x)) < 1e-10 or float(np.std(y)) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(
    snapshots: list[dict[str, Any]],
    predictions: list[np.ndarray],
    *,
    sampling_seed: int,
) -> dict[str, Any]:
    if len(snapshots) < 4:
        raise ValueError("At least four scenes required")
    if len({s["seed"] for s in snapshots}) != len(snapshots):
        raise ValueError("Scene seeds must be distinct")
    fixed_qpos = snapshots[0]["actual_qpos"]
    first = predictions[0]
    scene_rows = []
    for scene, actions in zip(snapshots, predictions, strict=True):
        if actions.shape != (10, 8):
            raise ValueError(f"Invalid 10x8 policy output for seed {scene['seed']}")
        scene_rows.append({
            "seed": int(scene["seed"]),
            "cubeA_xyz_m": scene["cubeA_xyz_m"].tolist(),
            "cubeB_xyz_m": scene["cubeB_xyz_m"].tolist(),
            "tcp_xyz_m": scene["tcp_xyz_m"].tolist(),
            "initial_tcp_to_cubeA_m": float(
                np.linalg.norm(scene["tcp_xyz_m"] - scene["cubeA_xyz_m"])
            ),
            "actual_robot_qpos_l2_from_reference": float(
                np.linalg.norm(scene["actual_qpos"] - fixed_qpos)
            ),
            "camera_mae_vs_reference_0_to_255": float(np.abs(
                snapshots[0]["image"].astype(np.float32)
                - scene["image"].astype(np.float32)
            ).mean()),
            "first_predicted_action_8d": actions[0].tolist(),
            "last_predicted_action_8d": actions[-1].tolist(),
            "mean_gripper_command_10": float(actions[:, 7].mean()),
            "first_arm_target_shift_from_fixed_qpos_l2_rad": float(
                np.linalg.norm(actions[0, :7] - fixed_qpos[:7])
            ),
            "arm_mae_10_vs_reference_rad": compare_actions(
                first, actions
            )["arm_mae_all_10_actions_rad"],
        })

    pairs = pairwise_metrics(snapshots, predictions)
    red_xy = np.stack([s["cubeA_xyz_m"][:2] for s in snapshots])
    green_xy = np.stack([s["cubeB_xyz_m"][:2] for s in snapshots])
    unique_positions = len({
        tuple(np.round(pos, 4)) for pos in red_xy
    })
    if unique_positions < 3:
        raise ValueError(
            "Fewer than three distinct red cube XY positions: "
            "increase the number/range of seeds"
        )
    qpos_distances = [s["actual_robot_qpos_l2_from_reference"] for s in scene_rows]
    return {
        "experiment": "fixed_pi05_qpos_and_flow_noise_scene_image_sensitivity",
        "interpretation_limit": (
            "Scene resets vary cube A/B and potentially rendered robot/background; "
            "measured action response does not establish movement toward the cube."
        ),
        "seeds": [s["seed"] for s in snapshots],
        "sampling_seed": int(sampling_seed),
        "fixed_model_qpos_9d": fixed_qpos.tolist(),
        "scene_count": len(snapshots),
        "distinct_cubeA_xy_positions": unique_positions,
        "cubeA_x_span_m": float(np.ptp(red_xy[:, 0])),
        "cubeA_y_span_m": float(np.ptp(red_xy[:, 1])),
        "cubeB_x_span_m": float(np.ptp(green_xy[:, 0])),
        "cubeB_y_span_m": float(np.ptp(green_xy[:, 1])),
        "max_actual_qpos_deviation_l2": float(max(qpos_distances)),
        "mean_pairwise_cubeA_xy_separation_m": float(np.mean(
            [p["cubeA_xy_separation_m"] for p in pairs]
        )),
        "mean_pairwise_arm_chunk_mae_rad": float(np.mean(
            [p["arm_mae_all_10_actions_rad"] for p in pairs]
        )),
        "mean_pairwise_gripper_chunk_mae": float(np.mean(
            [p["gripper_mae_all_10_actions"] for p in pairs]
        )),
        "pairwise_cubeA_distance_vs_arm_action_mae_pearson_descriptive":
            descriptive_pairwise_corr(pairs),
        "scenes": scene_rows,
        "pairs": pairs,
    }


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch

    from rlinf.models.embodiment.openpi import get_model

    if len(args.seeds) < 4 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide at least four distinct --seeds")
    ckpt = args.checkpoint.expanduser().resolve()
    norm_stats = args.norm_stats.expanduser().resolve()
    if not (ckpt / "model_state_dict/full_weights.pt").is_file():
        raise FileNotFoundError(f"Missing trained model weights in {ckpt}")
    if not norm_stats.is_file():
        raise FileNotFoundError(f"Missing norm stats: {norm_stats}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Pi0.5 inference")

    model_cfg, _ = evaluator.make_model_config(
        args.config, args.model_config, ckpt, norm_stats
    )
    model = get_model(model_cfg).to("cuda:0").eval()

    env = gym.make(
        "StackCube-v1",
        robot_uids="panda", num_envs=1, obs_mode="rgb",
        control_mode="pd_joint_pos", reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 200, "control_freq": 20},
        sensor_configs={"width": 224, "height": 224},
        max_episode_steps=300,
    )
    try:
        snapshots = [capture_scene(env, seed) for seed in args.seeds]
        fixed_qpos = snapshots[0]["actual_qpos"]
        with torch.inference_mode():
            predictions = [
                infer(model, scene["image"], fixed_qpos, seed=args.sampling_seed)
                for scene in snapshots
            ]
            repeat = infer(
                model, snapshots[0]["image"], fixed_qpos,
                seed=args.sampling_seed
            )
        repeat_diff = compare_actions(predictions[0], repeat)
        summary = summarize(
            snapshots, predictions, sampling_seed=args.sampling_seed
        )
        summary["checkpoint"] = str(ckpt)
        summary["normalization_stats"] = str(norm_stats)
        summary["repeat_same_input"] = repeat_diff

        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        if args.save_images:
            import imageio.v2 as imageio

            for scene in snapshots:
                imageio.imwrite(
                    str(output / f"seed_{scene['seed']}_base_camera.png"),
                    scene["image"],
                )
        json_path = output / "object_generalization.json"
        json_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

        print(
            "OBJECT GENERALIZATION: "
            f"scenes={summary['scene_count']} "
            f"distinct_red_positions={summary['distinct_cubeA_xy_positions']} "
            f"red_x_span_m={summary['cubeA_x_span_m']:.3f} "
            f"red_y_span_m={summary['cubeA_y_span_m']:.3f}"
        )
        print(
            "POLICY RESPONSE: "
            f"pair_red_xy_m={summary['mean_pairwise_cubeA_xy_separation_m']:.3f} "
            f"pair_arm_mae_rad={summary['mean_pairwise_arm_chunk_mae_rad']:.5f} "
            f"pair_gripper_mae={summary['mean_pairwise_gripper_chunk_mae']:.5f} "
            f"repeat_arm_mae_rad={repeat_diff['arm_mae_all_10_actions_rad']:.5f}"
        )
        corr = summary["pairwise_cubeA_distance_vs_arm_action_mae_pearson_descriptive"]
        print(
            "LIMITS: "
            f"pairwise_corr_descriptive={corr if corr is not None else 'undefined'} "
            f"max_actual_qpos_change_l2={summary['max_actual_qpos_deviation_l2']:.5f} "
            "(not a closed-loop success test)"
        )
        print(f"Saved: {json_path}")
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
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS)
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
