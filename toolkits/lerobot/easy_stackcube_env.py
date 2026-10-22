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

"""Easy, narrowly randomized StackCube-v1 with a real Panda wrist camera.

Import this module once before gym.make("StackCubeEasyTwoCam-v1").
All stacking physics, rewards, task success rules and robot controllers are
inherited unchanged from ManiSkill's StackCubeEnv. Only object initialization
is simplified. It does NOT override global ManiSkill environments.
"""

from __future__ import annotations

import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

ENV_ID = "StackCubeEasyTwoCam-v1"
ROBOT_UID = "panda_wristcam"

# XY meters, chosen well within the official StackCube tabletop spawn region.
RED_CENTER_XY = (0.0, -0.10)
GREEN_CENTER_XY = (0.0, 0.10)
XY_HALF_WIDTH_M = 0.025  # +/-2.5cm per axis around each center
CUBE_Z_M = 0.02  # 4cm cube resting on the tabletop
MAX_YAW_RAD = 0.15


def sample_cube_poses(
    batch_size: int, device: torch.device | str, *, xy_half_width_m: float = XY_HALF_WIDTH_M
):
    """Separate, non-overlapping object regions with small seeded perturbations.

    Called inside ManiSkill's seeded reset. Uses torch RNG with no custom seed.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with torch.device(device):
        delta = (torch.rand((2, batch_size, 2)) * 2 - 1) * xy_half_width_m
        yaw = (torch.rand((2, batch_size)) * 2 - 1) * MAX_YAW_RAD
        positions = torch.zeros((2, batch_size, 3))
        positions[0, :, :2] = torch.tensor(RED_CENTER_XY) + delta[0]
        positions[1, :, :2] = torch.tensor(GREEN_CENTER_XY) + delta[1]
        positions[:, :, 2] = CUBE_Z_M
        quaternion = torch.zeros((2, batch_size, 4))
        quaternion[:, :, 0] = torch.cos(yaw / 2)
        quaternion[:, :, 3] = torch.sin(yaw / 2)
        return positions, quaternion


@register_env(ENV_ID, max_episode_steps=300)
class EasyTwoCameraStackCubeEnv(StackCubeEnv):
    """Narrow-position StackCube; use panda_wristcam and 224px RGB in gym.make."""

    def __init__(self, *args, robot_uids=ROBOT_UID, robot_init_qpos_noise=0.0, **kwargs):
        if robot_uids != ROBOT_UID:
            raise ValueError(f"{ENV_ID} requires robot_uids={ROBOT_UID!r}")
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        positions, quaternions = sample_cube_poses(len(env_idx), self.device)
        self.cubeA.set_pose(Pose.create_from_pq(p=positions[0], q=quaternions[0]))
        self.cubeB.set_pose(Pose.create_from_pq(p=positions[1], q=quaternions[1]))

# New controlled experiment: EE delta-pose actions, narrower XY randomization.
# Register separately so the previously prepared 2.5cm-joint baseline is unchanged.
EE_ENV_ID = "StackCubeEasyEEDeltaTwoCam-v1"
EE_XY_HALF_WIDTH_M = 0.01  # independent +/-1.0 cm for both cubes


@register_env(EE_ENV_ID, max_episode_steps=300)
class EasyEEDeltaTwoCameraStackCubeEnv(EasyTwoCameraStackCubeEnv):
    """Same stacking physics, real wrist camera and fixed robot start, +/-1 cm."""

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        positions, quaternions = sample_cube_poses(
            len(env_idx), self.device, xy_half_width_m=EE_XY_HALF_WIDTH_M
        )
        self.cubeA.set_pose(Pose.create_from_pq(p=positions[0], q=quaternions[0]))
        self.cubeB.set_pose(Pose.create_from_pq(p=positions[1], q=quaternions[1]))
