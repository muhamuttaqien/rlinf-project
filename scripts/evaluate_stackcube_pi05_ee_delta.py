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

"""Evaluate real-two-camera Pi0.5 on native 7D EE delta StackCube.

Uses 1 action per 10-step prediction (replan every simulator control step).
No joint-position actions, state estimates from privileged cube coordinates,
or expert assistance are used. Success comes only from ManiSkill.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.evaluate_stackcube_pi05 import (
    as_numpy, make_model_config, scalar_bool, success_from_info, save_video
)
from toolkits.lerobot.stackcube_ee_delta import (
    ACTION_DIM, extract_ee_observation, make_env, policy_input
)

CONFIG = Path("examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_easy_ee_delta_two_cam.yaml")
MODEL_CONFIG = Path("examples/sft/config/model/pi0_5.yaml")
DEFAULT_STATS = Path(
    "/data1/Muttaqien/rlinf_data/pi05_stackcube_easy_ee_delta_two_cam_assets/"
    "local/stackcube_easy_ee_delta_two_cam100/norm_stats.json"
)
DEFAULT_OUTPUT = Path(
    "/data1/Muttaqien/rlinf_data/results/"
    "maniskill_stackcube_easy_ee_delta_two_cam_pi05_state_sft/evaluation"
)


def decode_action_chunk(predicted, *, action_chunk: int = 10) -> np.ndarray:
    """Return environment-space 7D EE actions, preserving order and time."""
    if action_chunk not in (1, 10):
        raise ValueError("action_chunk must be 1 or 10")
    actions = as_numpy(predicted)
    if (
        actions.ndim != 3 or actions.shape[0] != 1
        or actions.shape[1] < action_chunk or actions.shape[2] != ACTION_DIM
        or not np.isfinite(actions).all()
    ):
        raise ValueError(
            f"Expected finite [1,horizon,{ACTION_DIM}] EE actions with "
            f"horizon >= {action_chunk}, got {actions.shape}"
        )
    return np.asarray(actions[0, :action_chunk], dtype=np.float32)


def decode_first_action(predicted) -> np.ndarray:
    """Legacy helper for first-action replanning."""
    return decode_action_chunk(predicted, action_chunk=1)[0]


def _robot_grasped(env, info) -> bool | None:
    """Use robot contact detection when available; don't invent 0 grasps."""
    agent = env.unwrapped.agent
    if hasattr(agent, "is_grasping"):
        return scalar_bool(agent.is_grasping(env.unwrapped.cubeA))
    if "is_cubeA_grasped" in info:
        return scalar_bool(info["is_cubeA_grasped"])
    return None


def evaluate_one(
    env, model, seed: int, *, max_steps: int, record_video: bool,
    action_chunk: int = 10,
):
    """Evaluate executable chunks and report initial command for investigation."""
    if action_chunk not in (1, 10):
        raise ValueError("action_chunk must be 1 or 10")
    obs, _ = env.reset(seed=seed)
    steps = 0
    success = False
    grasped: bool | None = None
    closest = float("inf")
    frames = []
    first_action = None
    first_tcp = as_numpy(env.unwrapped.agent.tcp.pose.p).reshape(3).copy()
    first_cube = as_numpy(env.unwrapped.cubeA.pose.p).reshape(3).copy()
    finished = False

    while steps < max_steps and not finished:
        parsed = extract_ee_observation(env, obs)
        if record_video and (steps % 2 == 0 or not frames):
            frames.append(parsed["image"].copy())
        action_batch, _ = model.predict_action_batch(
            policy_input(parsed), mode="eval"
        )
        actions = decode_action_chunk(action_batch, action_chunk=action_chunk)
        if first_action is None:
            first_action = actions[0].tolist()
        for action in actions:
            obs, _reward, terminated, truncated, info = env.step(action[None, :])
            steps += 1
            success |= success_from_info(info)
            contact = _robot_grasped(env, info)
            if contact is not None:
                grasped = bool(grasped) or contact
            tcp = as_numpy(env.unwrapped.agent.tcp.pose.p).reshape(3)
            cube = as_numpy(env.unwrapped.cubeA.pose.p).reshape(3)
            closest = min(closest, float(np.linalg.norm(tcp - cube)))
            finished = (
                success or scalar_bool(terminated) or scalar_bool(truncated)
                or steps >= max_steps
            )
            if record_video and (steps % 2 == 0 or finished):
                frames.append(extract_ee_observation(env, obs)["image"].copy())
            if finished:
                break
    if record_video and not frames:
        frames.append(extract_ee_observation(env, obs)["image"].copy())
    return {
        "seed": seed, "success": success, "steps": steps,
        "ever_grasped": grasped, "min_tcp_to_red_cube_m": closest,
        "initial_tcp_xyz": first_tcp.tolist(),
        "initial_red_cube_xyz": first_cube.tolist(),
        "first_ee_action": first_action,
        "action_chunk_executed": action_chunk,
    }, frames

def run(args: argparse.Namespace):
    import torch
    from rlinf.models.embodiment.openpi import get_model

    if args.action_chunk not in (1, 10):
        raise ValueError('--action-chunk must be 1 or 10')
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("--episodes and --max-steps must be positive")
    ckpt = args.checkpoint.expanduser().resolve()
    stats = args.norm_stats.expanduser().resolve()
    if not (ckpt / "model_state_dict/full_weights.pt").is_file():
        raise FileNotFoundError(f"Missing trained 7D EE checkpoint in {ckpt}")
    if not stats.is_file():
        raise FileNotFoundError(f"Missing new 8D/7D norm stats at {stats}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    cfg, _ = make_model_config(args.config, MODEL_CONFIG, ckpt, stats)
    if str(cfg.openpi.config_name) != "pi05_stackcube_panda_easy_ee_delta_two_cam_state":
        raise ValueError("This evaluator requires the EE-delta two-camera model")

    model = get_model(cfg).to("cuda:0").eval()
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(max_episode_steps=args.max_steps)
    results = []
    try:
        with torch.inference_mode():
            for i in range(args.episodes):
                seed = args.seed_start + i
                result, frames = evaluate_one(
                    env, model, seed, max_steps=args.max_steps,
                    record_video=(i == 0), action_chunk=args.action_chunk,
                )
                results.append(result)
                if frames:
                    save_video(
                        frames, out / f"seed_{seed:05d}.mp4", fps=10
                    )
                print(
                    f"EPISODE seed={seed} success={int(result['success'])} "
                    f"grasped={result['ever_grasped']} "
                    f"min_tcp_m={result['min_tcp_to_red_cube_m']:.3f} "
                    f"first_action={np.round(result['first_ee_action'], 3)}"
                )
    finally:
        env.close()

    successes = sum(r["success"] for r in results)
    report = {
        "experiment": "easy_ee_delta_two_camera",
        "config": str(args.config),
        "checkpoint": str(ckpt),
        "action_dim": 7, "state_dim": 8, "cameras": 2,
        "action_chunk_executed": args.action_chunk,
        "episodes": len(results), "successes": successes,
        "success_rate": successes / len(results), "per_episode": results,
    }
    (out / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"RESULT: {successes}/{len(results)} successful; "
        f"grasped={sum(r['ever_grasped'] is True for r in results)}/{len(results)}; "
        f"saved {out / 'results.json'}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--norm-stats", type=Path,
        default=Path(os.getenv("STACKCUBE_NORM_STATS", DEFAULT_STATS)),
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--action-chunk", type=int, choices=(1, 10), default=10,
        help="Execute 10 actions per prediction, or 1 to replan each step",
    )
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
