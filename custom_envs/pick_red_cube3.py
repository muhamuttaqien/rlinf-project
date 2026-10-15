"""Three colored cubes, fixed red target, with a lift-only SAC task."""

import sapien
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PickRedCube3-v1", max_episode_steps=100)
class PickRedCube3Env(PickCubeEnv):
    """Pick and lift the red cube; the green and blue cubes are distractors."""

    lift_height = 0.10

    def _load_scene(self, options):
        super()._load_scene(options)
        # PickCubeEnv already creates a physical red cube as self.cube.
        self.green_cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size,
            color=[0.05, 0.85, 0.10, 1.0], name="green_cube",
            initial_pose=sapien.Pose(p=[0, 0.17, self.cube_half_size]),
        )
        self.blue_cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size,
            color=[0.1, 0.25, 1.0, 1.0], name="blue_cube",
            initial_pose=sapien.Pose(p=[0, -0.17, self.cube_half_size]),
        )

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        # Keep the parent task's observation fields, but set its visible goal
        # marker at the target lift position rather than a random place goal.
        with torch.device(self.device):
            b = len(env_idx)
            red_xyz = self.cube.pose.p[env_idx].clone()
            red_xyz[:, 2] = self.cube_half_size + self.lift_height
            self.goal_site.set_pose(Pose.create_from_pq(red_xyz))
            for cube, y in ((self.green_cube, 0.17), (self.blue_cube, -0.17)):
                xyz = torch.zeros((b, 3))
                xyz[:, 0] = (torch.rand(b) - 0.5) * 0.05
                xyz[:, 1] = y + (torch.rand(b) - 0.5) * 0.04
                xyz[:, 2] = self.cube_half_size
                cube.set_pose(Pose.create_from_pq(xyz))

    def evaluate(self):
        red_grasped = self.agent.is_grasping(self.cube)
        green_grasped = self.agent.is_grasping(self.green_cube)
        blue_grasped = self.agent.is_grasping(self.blue_cube)
        wrong_grasp = green_grasped | blue_grasped
        red_height = self.cube.pose.p[:, 2] - self.cube_half_size
        red_lifted = red_height >= self.lift_height
        success = red_grasped & red_lifted & ~wrong_grasp
        return {
            "success": success,
            "is_grasped": red_grasped,  # inherited PickCube obs expects this
            "is_obj_placed": red_lifted,  # preserves compatible logging keys
            "red_lifted": red_lifted,
            "wrong_grasp": wrong_grasp,
        }

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            obs["green_pos"] = self.green_cube.pose.p
            obs["blue_pos"] = self.blue_cube.pose.p
        return obs

    def compute_dense_reward(self, obs, action, info):
        reach_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, dim=1
        )
        reach_reward = 1 - torch.tanh(5 * reach_dist)
        grasp_reward = info["is_grasped"].float()
        height = (self.cube.pose.p[:, 2] - self.cube_half_size).clamp(min=0)
        lift_reward = (height / self.lift_height).clamp(max=1) * grasp_reward
        wrong_penalty = 2.0 * info["wrong_grasp"].float()
        reward = reach_reward + grasp_reward + 2.0 * lift_reward - wrong_penalty
        return torch.where(info["success"], torch.full_like(reward, 5.0), reward)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
