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


def decode_first_action(predicted) -> np.ndarray:
    actions = as_numpy(predicted)
    if (
        actions.ndim != 3 or actions.shape[0] != 1
        or actions.shape[1] < 1 or actions.shape[2] != ACTION_DIM
        or not np.isfinite(actions).all()
    ):
        raise ValueError(f"Expected finite [1,horizon,{ACTION_DIM}] EE actions, got {actions.shape}")
    return np.asarray(actions[0, 0], dtype=np.float32)


def evaluate_one(env, model, seed: int, *, max_steps: int, record_video: bool):
    obs, _ = env.reset(seed=seed)
    steps = 0
    success = False
    grasped = False
    closest = float("inf")
    frames = []
    while steps < max_steps:
        parsed = extract_ee_observation(env, obs)
        if record_video and steps % 2 == 0:
            frames.append(parsed["image"].copy())
        action_chunk, _ = model.predict_action_batch(
            policy_input(parsed), mode="eval"
        )
        # This action already lives in ManiSkill's normalized 7D
        # pd_ee_delta_pose action space. Do NOT apply delta transforms twice.
        action = decode_first_action(action_chunk)
        obs, _reward, terminated, truncated, info = env.step(action[None, :])
        steps += 1
        success |= success_from_info(info)
        if "is_cubeA_grasped" in info:
            grasped |= scalar_bool(info["is_cubeA_grasped"])
        tcp = as_numpy(env.unwrapped.agent.tcp.pose.p).reshape(3)
        cube = as_numpy(env.unwrapped.cubeA.pose.p).reshape(3)
        closest = min(closest, float(np.linalg.norm(tcp - cube)))
        if success or scalar_bool(terminated) or scalar_bool(truncated):
            break
    if record_video:
        frames.append(extract_ee_observation(env, obs)["image"].copy())
    return {
        "seed": seed, "success": success, "steps": steps,
        "ever_grasped": grasped, "min_tcp_to_red_cube_m": closest,
    }, frames


def run(args: argparse.Namespace):
    import torch
    from rlinf.models.embodiment.openpi import get_model

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
                    env, model, seed, max_steps=args.max_steps, record_video=(i == 0)
                )
                results.append(result)
                if frames:
                    save_video(
                        frames, out / f"seed_{seed:05d}.mp4", fps=10
                    )
                print(
                    f"EPISODE seed={seed} success={int(result['success'])} "
                    f"grasped={int(result['ever_grasped'])} "
                    f"min_tcp_m={result['min_tcp_to_red_cube_m']:.3f}"
                )
    finally:
        env.close()

    successes = sum(r["success"] for r in results)
    report = {
        "experiment": "easy_ee_delta_two_camera",
        "config": str(args.config),
        "checkpoint": str(ckpt),
        "action_dim": 7, "state_dim": 8, "cameras": 2,
        "action_chunk_executed": 1,
        "episodes": len(results), "successes": successes,
        "success_rate": successes / len(results), "per_episode": results,
    }
    (out / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"RESULT: {successes}/{len(results)} successful; "
        f"grasped={sum(r['ever_grasped'] for r in results)}/{len(results)}; "
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
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
