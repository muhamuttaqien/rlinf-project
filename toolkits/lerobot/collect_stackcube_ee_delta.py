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

"""Collect successful *native* 7D pd_ee_delta_pose StackCube demonstrations.

Unlike the existing StackCube joint-position planner, this expert commands
the actual EE delta controller. It uses ManiSkill's official OBB-based Panda
grasp geometry but issues pose-delta actions with feedback at every step.
Only ManiSkill-verified SUCCESS episodes are written. Stop and debug if too
few successes: never relabel unverified demonstrations as expert data.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import sapien

from toolkits.lerobot.collect_maniskill_stackcube_recap import _dataset_path
from rlinf.data.storage.lerobot import add_frame_to_dataset
from toolkits.lerobot.stackcube_ee_delta import (
    TASK, ACTION_DIM, STATE_DIM, extract_ee_observation, make_env,
    normalized_ee_action, pose_in_robot_root, root_relative_tcp,
)

LOG = logging.getLogger(__name__)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def collect_episode(env: Any, seed: int, *, max_steps: int = 300):
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb, get_actor_obb,
    )

    obs, _info = env.reset(seed=seed)
    base = env.unwrapped
    if base.control_mode != "pd_ee_delta_pose":
        raise ValueError("7D expert must run in pd_ee_delta_pose controller")
    if int(np.prod(env.single_action_space.shape)) != ACTION_DIM:
        raise ValueError(f"Expected 7D action space, got {env.single_action_space}")
    rows: list[dict] = []
    last_success = False
    done = False

    def apply(action: np.ndarray):
        nonlocal obs, last_success, done
        if len(rows) >= max_steps or done:
            return False
        parsed = extract_ee_observation(env, obs)
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite delta EE expert command")
        next_obs, _reward, terminated, truncated, info = env.step(action[None])
        rows.append({
            **parsed,
            "actions": action,
            "is_success": np.array([True], dtype=np.bool_),
            "task": TASK,
        })
        obs = next_obs
        last_success = bool(_as_numpy(info["success"]).reshape(-1)[0])
        done = last_success or bool(_as_numpy(terminated).reshape(-1)[0]) or bool(
            _as_numpy(truncated).reshape(-1)[0]
        )
        return True

    def gripper(command: float, steps: int):
        for _ in range(steps):
            if not apply(np.array([0, 0, 0, 0, 0, 0, command], dtype=np.float32)):
                break

    def move(world_target: sapien.Pose, *, gripper_command: float = 1.0, limit: int = 60):
        target_p, target_q = pose_in_robot_root(env, world_target)
        for _ in range(limit):
            if done or len(rows) >= max_steps:
                return False
            p, q = root_relative_tcp(env)
            quat_err = abs(float(np.dot(q, target_q)))
            if np.linalg.norm(target_p - p) < 0.009 and quat_err > 0.995:
                return True
            action = normalized_ee_action(
                p, q, target_p, target_q, gripper_command
            )
            if not apply(action):
                return False
        return False

    try:
        obb = get_actor_obb(base.cubeA)
        approach = np.array([0, 0, -1.0], dtype=float)
        closing = _as_numpy(
            base.agent.tcp.pose.to_transformation_matrix()
        ).reshape(4, 4)[:3, 1]
        geometry = compute_grasp_info_by_obb(
            obb, approaching=approach, target_closing=closing, depth=0.025
        )
        grasp_pose = base.agent.build_grasp_pose(
            approach, geometry["closing"], geometry["center"]
        )
        if not move(grasp_pose * sapien.Pose([0, 0, -0.05])):
            return None
        if not move(grasp_pose):
            return None
        gripper(-1.0, 14)
        if done:
            return rows if last_success else None

        lift_pose = sapien.Pose([0, 0, 0.10]) * grasp_pose
        if not move(lift_pose, gripper_command=-1.0):
            return None
        target_red = _as_numpy(base.cubeB.pose.p).reshape(3) + np.array([0, 0, 0.04])
        # Keep actual grasp displacement after lifting; align the carried red cube
        # with the green cube. The 0.04m offset is one cube height.
        offset = target_red - _as_numpy(base.cubeA.pose.p).reshape(3)
        align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
        if not move(align_pose, gripper_command=-1.0):
            return None
        gripper(1.0, 14)
        gripper(1.0, 12)  # give cube time to become static after release
        return rows if last_success else None
    except (RuntimeError, ValueError, AssertionError, AttributeError, TypeError) as exc:
        LOG.warning("EE expert seed=%d failed: %s", seed, exc)
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="local/stackcube_easy_ee_delta_two_cam100")
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--max-attempts", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    if args.num_episodes < 1 or args.max_attempts < args.num_episodes:
        raise ValueError("Require max_attempts >= num_episodes > 0")
    target_dir = _dataset_path(args.repo_id)
    if target_dir.exists():
        raise FileExistsError(
            f"Dataset already exists at {target_dir}. "
            "Choose a fresh --repo-id; this collector never overwrites."
        )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    env = make_env()
    dataset = None
    success_count = 0
    try:
        for attempt in range(args.max_attempts):
            seed = args.seed + attempt
            frames = collect_episode(env, seed)
            if not frames:
                if attempt % 25 == 0:
                    LOG.info("attempts=%d successes=%d", attempt + 1, success_count)
                continue
            if dataset is None:
                dataset = LeRobotDataset.create(
                    repo_id=args.repo_id,
                    robot_type="panda",
                    fps=20,
                    features={
                        "image": {
                            "dtype": "image", "shape": (224, 224, 3),
                            "names": ["height", "width", "channel"],
                        },
                        "wrist_image": {
                            "dtype": "image", "shape": (224, 224, 3),
                            "names": ["height", "width", "channel"],
                        },
                        "state": {
                            "dtype": "float32", "shape": (STATE_DIM,),
                            "names": ["state"],
                        },
                        "actions": {
                            "dtype": "float32", "shape": (ACTION_DIM,),
                            "names": ["actions"],
                        },
                        "is_success": {
                            "dtype": "bool", "shape": (1,),
                            "names": ["is_success"],
                        },
                    },
                    image_writer_threads=2,
                    image_writer_processes=0,
                )
            for frame in frames:
                add_frame_to_dataset(dataset, frame)
            dataset.save_episode()
            success_count += 1
            LOG.info(
                "SAVED successful EE episode %d/%d seed=%d steps=%d",
                success_count, args.num_episodes, seed, len(frames)
            )
            if success_count >= args.num_episodes:
                break
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        env.close()
    if success_count != args.num_episodes:
        raise RuntimeError(
            f"Only {success_count}/{args.num_episodes} successful EE episodes "
            f"after {args.max_attempts} attempts; DO NOT train on this dataset."
        )
    print(f"COLLECTION PASSED: {success_count} successful 7D EE delta episodes")
    print(f"Saved: {target_dir}")


if __name__ == "__main__":
    main()
