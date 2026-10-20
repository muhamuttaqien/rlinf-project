# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Collect schema-inspection rollouts for Panda StackCube (not LeRobot/RECAP data)."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np


def to_numpy(value):
    """Convert tensor-like values without importing a specific accelerator backend."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def describe(value):
    """Summarize observation structure without dumping image tensors."""
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    array = to_numpy(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def scalar(value):
    """Extract a scalar from ManiSkill's single-environment tensor."""
    array = to_numpy(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected a single-env value, got shape {array.shape}")
    return array.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/stackcube_recap_probe"))
    parser.add_argument("--robot", choices=("panda", "panda_wristcam"), default="panda_wristcam")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        "StackCube-v1",
        robot_uids=args.robot,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        num_envs=1,
        render_mode="rgb_array",
    )
    summary = []
    rng = np.random.default_rng(args.seed)
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            if episode == 0:
                schema = {
                    "environment": "StackCube-v1",
                    "robot": args.robot,
                    "observation": describe(observation),
                    "action_space": str(env.action_space),
                    "control_mode": "pd_ee_delta_pose",
                    "note": "Random action probe only; NOT demonstrations or LeRobot data.",
                }
                (args.output / "schema.json").write_text(json.dumps(schema, indent=2))
            actions, rewards = [], []
            success = False
            for _ in range(args.max_steps):
                action = rng.uniform(
                    env.action_space.low, env.action_space.high
                ).astype(np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                actions.append(action)
                rewards.append(float(scalar(reward)))
                success = success or bool(scalar(info.get("success", False)))
                if bool(scalar(terminated)) or bool(scalar(truncated)):
                    break
            np.savez_compressed(
                args.output / f"probe_{episode:04d}.npz",
                actions=np.asarray(actions, dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                success=np.asarray(success, dtype=np.bool_),
            )
            summary.append({"episode": episode, "steps": len(actions), "success": success})
            print(f"episode={episode} steps={len(actions)} success={success}")
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Wrote inspection output to {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
