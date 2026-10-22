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

"""EE-delta StackCube I/O: two RGB images, 8D EE state, 7D controller action.

State [TCP world XYZ, TCP world quaternion WXYZ, finger-opening width].
Actions [normalized ΔXYZ, normalized ΔEulerXYZ, normalized gripper target]
are ManiSkill panda_wristcam/pd_ee_delta_pose controller commands.
The controller maps normalized translations to +/-0.1 m and normalized
rotations to Euler XYZ via +/-0.1 rad; rotation scale is negative in the
ManiSkill controller implementation. No joint-target / EE-action mixing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from toolkits.lerobot.collect_maniskill_stackcube_recap import (
    _extract_observation,
    _to_numpy,
)

STATE_DIM = 8
ACTION_DIM = 7
TASK = "stack the red cube on top of the green cube and release it"
ENV_ID = "StackCubeEasyEEDeltaTwoCam-v1"
ROBOT_UID = "panda_wristcam"
CONTROL_MODE = "pd_ee_delta_pose"


def ee_state(env: Any, obs: dict[str, Any]) -> np.ndarray:
    """World-frame TCP pose (7D) plus measured total finger opening (1D)."""
    base = env.unwrapped
    pose = base.agent.tcp.pose
    p = _to_numpy(pose.p).reshape(-1)
    q = _to_numpy(pose.q).reshape(-1)
    qpos = _to_numpy(obs["agent"]["qpos"]).reshape(-1)
    if p.shape != (3,) or q.shape != (4,) or qpos.shape != (9,):
        raise ValueError(f"Invalid Panda TCP/qpos shapes: {p.shape}, {q.shape}, {qpos.shape}")
    width = float(np.sum(qpos[7:9]))
    state = np.concatenate([p, q, [width]]).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite TCP or gripper state")
    return state


def extract_ee_observation(env: Any, obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """The same camera/state preprocessing for expert collection and evaluation."""
    original = _extract_observation(obs, require_wrist_camera=True)
    image, wrist = original["image"], original["wrist_image"]
    if image.shape != (224, 224, 3) or wrist.shape != (224, 224, 3):
        raise ValueError("Two 224x224 RGB cameras are required")
    return {
        "image": image,
        "wrist_image": wrist,
        "state": ee_state(env, obs),
    }


def policy_input(record: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "main_images": record["image"][None, ...],
        "wrist_images": record["wrist_image"][None, ...],
        "states": record["state"][None, ...],
        "task_descriptions": [TASK],
    }


def normalized_ee_action(
    current_p: np.ndarray,
    current_q_wxyz: np.ndarray,
    target_p: np.ndarray,
    target_q_wxyz: np.ndarray,
    gripper: float,
) -> np.ndarray:
    """Calculate normalized Panda EE delta-pose command in its root frame.

    Inputs must be end-effector poses *relative to the robot root*.
    ManiSkill's version used here composes target_q = delta_q * current_q,
    where delta_q is EulerXYZ and normalized rotation is scaled by -0.1.
    """
    from scipy.spatial.transform import Rotation

    p0 = np.asarray(current_p, dtype=np.float64).reshape(3)
    p1 = np.asarray(target_p, dtype=np.float64).reshape(3)
    q0 = np.asarray(current_q_wxyz, dtype=np.float64).reshape(4)
    q1 = np.asarray(target_q_wxyz, dtype=np.float64).reshape(4)
    delta_xyz = np.clip((p1 - p0) / 0.1, -1.0, 1.0)
    r0 = Rotation.from_quat(q0[[1, 2, 3, 0]])
    r1 = Rotation.from_quat(q1[[1, 2, 3, 0]])
    delta_rot = r1 * r0.inv()
    euler_xyz = delta_rot.as_euler("XYZ", degrees=False)
    # ManiSkill 3 current implementation multiplies normalized rotation by
    # rot_lower=-0.1; preserve sign for *this* controller version.
    rotation = (-euler_xyz / 0.1)
    norm = float(np.linalg.norm(rotation))
    if norm > 1.0:
        rotation /= norm
    action = np.concatenate([
        delta_xyz, rotation, [float(np.clip(gripper, -1.0, 1.0))]
    ]).astype(np.float32)
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError("Invalid normalized EE action")
    return action


def root_relative_tcp(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """ManiSkill EE controller computes deltas relative to robot's root."""
    root = env.unwrapped.agent.robot.root.pose
    ee = env.unwrapped.agent.tcp.pose
    local = root.inv() * ee
    return _to_numpy(local.p).reshape(3), _to_numpy(local.q).reshape(4)


def pose_in_robot_root(env: Any, world_pose: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert a desired world-space SAPIEN pose into robot-root coordinates."""
    from mani_skill.utils.structs.pose import Pose

    root = env.unwrapped.agent.robot.root.pose
    target = Pose.create(world_pose)
    relative = root.inv() * target
    return _to_numpy(relative.p).reshape(3), _to_numpy(relative.q).reshape(4)


def make_env(*, max_episode_steps: int = 300):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import toolkits.lerobot.easy_stackcube_env  # noqa: F401

    return gym.make(
        ENV_ID, robot_uids=ROBOT_UID, num_envs=1,
        obs_mode="rgb", control_mode=CONTROL_MODE, reward_mode="sparse",
        sim_backend="physx_cpu",
        sim_config={"sim_freq": 200, "control_freq": 20},
        sensor_configs={"width": 224, "height": 224},
        max_episode_steps=max_episode_steps,
    )
