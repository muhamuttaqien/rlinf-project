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

"""Check native Panda StackCube observations before preparing RECAP datasets.

Run from the repository root with the ManiSkill environment activated:
    python tests/e2e_tests/embodied/check_stackcube_panda.py --sim-backend cpu

This checks simulator inputs and outputs, not policy performance or RLinf's
OpenPI observation/action adapters. Zero arm deltas keep the reset pose.
"""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, found {value}")
    return parsed


def main() -> None:
    """Validate RGB, proprioception, actions, and native episode signals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=positive_int, default=10)
    parser.add_argument("--output", type=Path, help="Write a JSON check report")
    args = parser.parse_args()

    # Simulator dependencies are optional for other RLinf workloads.
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch

    env = gym.make(
        "StackCube-v1",
        robot_uids="panda",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        sim_backend=args.sim_backend,
        max_episode_steps=args.steps,
        sensor_configs={"width": 224, "height": 224},
    )
    try:
        obs, info = env.reset(seed=args.seed)

        def check_observation(observation: dict, step_info: dict) -> None:
            image = observation["sensor_data"]["base_camera"]["rgb"]
            if tuple(image.shape) != (1, 224, 224, 3) or image.dtype != torch.uint8:
                raise ValueError(f"Expected uint8 RGB [1,224,224,3], found {image.shape}, {image.dtype}")
            qpos = observation["agent"]["qpos"]
            if tuple(qpos.shape) != (1, 9) or not torch.isfinite(qpos).all():
                raise ValueError(f"Expected finite Panda qpos [1,9], found {qpos}")
            if "success" not in step_info:
                raise ValueError("Expected native StackCube success in environment info")

        check_observation(obs, info)
        action_shape = tuple(env.action_space.shape)
        if action_shape != (1, 7):
            raise ValueError(f"Expected batched Panda EE action [1,7], found {action_shape}")
        action = torch.zeros(action_shape, device=env.unwrapped.device)
        action[:, -1] = 1.0  # Keep the gripper open.
        completed_steps = 0
        success_once = False
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(action)
            check_observation(obs, info)
            if not torch.isfinite(reward).all():
                raise ValueError(f"Expected finite native reward, found {reward}")
            completed_steps += 1
            success_once |= bool(info["success"].any().item())
            if bool(terminated.any().item()) or bool(truncated.any().item()):
                break

        report = {
            "environment": "StackCube-v1",
            "robot": "panda",
            "mani_skill_version": version("mani_skill"),
            "torch_version": torch.__version__,
            "sim_backend": args.sim_backend,
            "seed": args.seed,
            "control_mode": "pd_ee_delta_pose",
            "action_shape": list(action_shape),
            "rgb_shape": list(obs["sensor_data"]["base_camera"]["rgb"].shape),
            "qpos_shape": list(obs["agent"]["qpos"].shape),
            "completed_steps": completed_steps,
            "native_success_once": success_once,
            "terminated": bool(terminated.any().item()),
            "truncated": bool(truncated.any().item()),
            "environment_check_passed": True,
        }
        serialized = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        # The report is the command's output, rather than a training log.
        sys.stdout.write(serialized)
    finally:
        env.close()


if __name__ == "__main__":
    main()
