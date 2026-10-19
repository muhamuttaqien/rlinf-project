# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Smoke-test ManiSkill's built-in Franka Panda StackCube task.

This verifies simulator registration, robot initialization, reset, and stepping.
It does not run an RLinf policy or a world model.
"""

import argparse

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="panda", choices=["panda", "panda_wristcam"])
    parser.add_argument("--obs-mode", default="state", choices=["state", "rgb"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")

    env = gym.make(
        "StackCube-v1",
        robot_uids=args.robot,
        obs_mode=args.obs_mode,
        num_envs=1,
        control_mode="pd_ee_delta_pose",
    )
    try:
        obs, info = env.reset(seed=args.seed)
        print(f"Environment: StackCube-v1 | Robot: {args.robot}")
        print(f"Observation mode: {args.obs_mode}")
        print(f"Action space: {env.action_space}")
        print(f"Initial observation type: {type(obs).__name__}")
        print(f"Initial info keys: {list(info)}")
        for step in range(args.steps):
            # Zero-command checks basic stepping without attempting the task.
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"step={step + 1} reward={reward} "
                f"terminated={terminated} truncated={truncated} "
                f"success={info.get('success', 'n/a')}"
            )
        print("Smoke test completed (task success is not expected).")
    finally:
        env.close()


if __name__ == "__main__":
    main()
