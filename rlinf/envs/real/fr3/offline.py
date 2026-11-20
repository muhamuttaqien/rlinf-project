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

"""Native FR3 environment integration with fixed synthetic resources only."""

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.real.env import RealWorldEnv, StepResult
from rlinf.envs.utils import to_tensor
from rlinf.robotics.parts.fr3_synthetic import (
    SyntheticCamera,
    SyntheticFR3Observation,
    SyntheticRobotiqObservation,
    SyntheticSpaceMouse,
)
from rlinf.robotics.parts.ros2_offline_action import (
    FR3OfflineAction,
    RobotiqOfflineAction,
)
from rlinf.robotics.parts.transports.ros2.action_contract import (
    ActionRejected,
    ActionRequest,
    MockAuthorization,
    MonotonicClock,
    ObservationProvenance,
)
from rlinf.robotics.robots.franka import FrankaRobot

from .config import (
    ENV_ID,
    STATE_FIELDS,
    mock_profiles,
    validate_environment_config,
    validate_task_config,
)

STATE_DIMENSIONS = {
    key: 7
    if key
    in ("arm_joint_position", "arm_joint_velocity", "arm_joint_effort", "tcp_pose")
    else 1
    if key in ("knuckle_position_rad", "normalized_closure")
    else 3
    for key in STATE_FIELDS
}


class FR3OfflineTask(gym.Env):
    """Compose native parts while requiring typed, explicitly approved mock actions.

    The Box describes the seven numerical payload channels only. step() requires
    ActionRequest envelopes; a Box sample or collector zero array is rejected.
    """

    TELEOP: tuple = ()

    def __init__(
        self,
        override_cfg: Any,
        worker_info: Any = None,
        robot_info: Any = None,
        env_idx: int = 0,
    ) -> None:
        if robot_info is not None or worker_info is not None or env_idx != 0:
            raise ValueError(
                "Offline FR3 does not accept allocated physical hardware or multiple envs"
            )
        self.config = validate_task_config(override_cfg)
        self.task_description = self.config.task_description
        self.clock = MonotonicClock()
        self.profiles = mock_profiles(self.config)
        self.actions = {
            "arm": FR3OfflineAction(
                controller_id=self.profiles["arm"].interface.controller_id,
                configuration_id="synthetic:fr3-config-v1",
                clock=self.clock,
            ),
            "end_effector": RobotiqOfflineAction(
                controller_id=self.profiles["end_effector"].interface.controller_id,
                configuration_id="synthetic:fr3-config-v1",
                clock=self.clock,
            ),
        }
        self.robot = FrankaRobot(
            arm_observation=SyntheticFR3Observation(),
            gripper_observation=SyntheticRobotiqObservation(),
            arm=self.actions["arm"],
            end_effector=self.actions["end_effector"],
            spacemouse=SyntheticSpaceMouse(),
            **{
                name: SyntheticCamera(*camera.fixture_resolution, sensor=name)
                for name, camera in self.config.cameras.items()
            },
        )
        self.action_space = gym.spaces.Box(
            np.array([-1.0] * 6 + [0.0], dtype=np.float32), np.ones(7, dtype=np.float32)
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
                        for key, size in STATE_DIMENSIONS.items()
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0,
                            255,
                            (
                                camera.fixture_resolution[1],
                                camera.fixture_resolution[0],
                                3,
                            ),
                            np.uint8,
                        )
                        for name, camera in self.config.cameras.items()
                    }
                ),
            }
        )
        self._last: dict[str, Any] | None = None
        self._episode_id = ""
        self._needs_reset = True
        self._steps = 0

    def _observe(self) -> tuple[dict[str, Any], dict[str, Any]]:
        raw = self.robot.get_observation()
        self._last = raw
        combined = {
            **raw["arm_observation"]["state"],
            **raw["gripper_observation"]["state"],
        }
        observation = {
            "state": {
                key: np.asarray(combined[key], dtype=np.float32).reshape(-1)
                for key in STATE_FIELDS
            },
            "frames": {name: raw[name]["image"] for name in self.config.cameras},
        }
        metadata = {
            "synthetic": True,
            "execution_mode": "offline_synthetic",
            "episode_id": self._episode_id,
            "qualification_gate": "NO_GO",
            "physical_command_authorization": "NOT_GRANTED",
            "physical_safety_qualification": "NOT_QUALIFIED",
            "eligible_real_demonstration": False,
            "reward_evaluated": False,
            "success_evaluated": False,
            "ros2": {
                "arm": raw["arm_observation"]["ros2"],
                "gripper": raw["gripper_observation"]["ros2"],
            },
            "camera_metadata": {
                name: raw[name]["metadata"] for name in self.config.cameras
            },
            "spacemouse_raw": raw["spacemouse"],
            "controller_acceptance": "UNAVAILABLE: mock only",
            "measured_robot_response": "UNAVAILABLE: fixture is independent of requested actions",
            "action_events": {
                name: part.events() for name, part in self.actions.items()
            },
        }
        return observation, metadata

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        """Create a new disarmed synthetic session; never reset physical hardware."""
        super().reset(seed=seed)
        self.close()
        try:
            self.robot.connect()
            self._episode_id = "synthetic:" + uuid.uuid4().hex
            self._steps = 0
            self._needs_reset = False
            return self._observe()
        except BaseException:
            self.close()
            raise

    def authorize_synthetic(self) -> None:
        """Explicitly renew mock-only approval, discarding previous mock requests."""
        if self._needs_reset:
            raise RuntimeError(
                "Reset the offline environment before mock authorization"
            )
        try:
            for part in self.actions.values():
                part.reset()
            for name, part in self.actions.items():
                profile = self.profiles[name]
                authorization = MockAuthorization(
                    "synthetic:" + uuid.uuid4().hex,
                    self.config.source_id,
                    part.session_id,
                    part.authority_id,
                    profile.profile_id,
                    self.clock.domain,
                    self.clock.now_ns() + 60_000_000_000,
                )
                part.arm_mock(profile, authorization)
        except BaseException:
            for part in self.actions.values():
                part.reset()
            raise

    def create_normalized_action(
        self, arm_values: tuple[float, ...], closure: float, *, sequence: int
    ) -> dict[str, ActionRequest]:
        """Create typed synthetic requests from explicit normalized payloads.

        This neither approves nor submits them. Provenance uses the last fixture
        observation; stale observations remain stale and can be rejected.
        """
        if self._last is None or self._needs_reset:
            raise RuntimeError("Reset the offline environment before creating requests")
        now = self.clock.now_ns()
        result = {}
        for name, values, observation_key in (
            ("arm", arm_values, "arm_observation"),
            ("end_effector", (closure,), "gripper_observation"),
        ):
            part = self.actions[name]
            provenance = ObservationProvenance.from_observation(
                self._last[observation_key],
                clock_domain=self.clock.domain,
                timing_uncertainty_ns=0,
            )
            result[name] = ActionRequest(
                part.KIND,
                values,
                self.config.source_id,
                part.session_id,
                part.authority_id,
                sequence,
                now,
                now + self.config.synthetic_profile.max_lifetime_ns,
                self.clock.domain,
                provenance,
                part.interface,
            )
        return result

    def step(
        self, action: Mapping[str, ActionRequest]
    ) -> tuple[dict, float, bool, bool, dict]:
        """Admit both envelopes before submitting either to fixed mock endpoints."""
        if self._needs_reset:
            raise RuntimeError("Reset the offline environment before stepping")
        try:
            if not isinstance(action, Mapping) or set(action) != set(self.actions):
                raise ActionRejected(
                    "action_shape",
                    "Expected arm/end_effector typed ActionRequest mappings",
                )
            for name, observation_key in (
                ("arm", "arm_observation"),
                ("end_effector", "gripper_observation"),
            ):
                request = action[name]
                if (
                    type(request) is not ActionRequest
                    or type(request.observation) is not ObservationProvenance
                ):
                    raise ActionRejected(
                        "action_type",
                        "Expected typed action and observation provenance",
                    )
                try:
                    snapshot = json.loads(request.observation.snapshot_json)
                except (TypeError, ValueError) as exc:
                    raise ActionRejected(
                        "observation", "Expected valid observation JSON"
                    ) from exc
                if snapshot != self._last[observation_key]:
                    raise ActionRejected(
                        "observation",
                        "Expected typed requests tied to this environment's synthetic observation",
                    )
            if action["arm"].sequence != action["end_effector"].sequence:
                raise ActionRejected("sequence", "Expected one shared pair sequence")
            for name, part in self.actions.items():
                part.queue_action(action[name])
            submissions = {
                name: part.submit_pending() for name, part in self.actions.items()
            }
            self._steps += 1
            observation, info = self._observe()
            info["mock_submissions"] = submissions
            return observation, 0.0, False, False, info
        except BaseException:
            self.robot.reset()
            self._needs_reset = True
            raise

    def close(self) -> None:
        """Close only the owned synthetic parts and stdlib mock workers."""
        self._needs_reset = True
        self.robot.disconnect()


class FR3OfflineRealWorldEnv(RealWorldEnv):
    """Reuse native batching/observation/metrics with an offline typed-action boundary."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int = 1,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info: Any = None,
    ) -> None:
        validate_environment_config(cfg)
        if (
            num_envs != 1
            or total_num_processes != 1
            or (worker_info is not None and getattr(worker_info, "hardware_infos", []))
        ):
            raise ValueError(
                "Expected one local synthetic environment with no physical allocation"
            )
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    @staticmethod
    def realworld_setup() -> None:
        """Perform no ROS, daemon, device or scheduler setup."""

    def _create_env(self, env_idx: int) -> gym.Env:
        return gym.make(
            ENV_ID,
            override_cfg=OmegaConf.to_container(self.cfg.override_cfg, resolve=True),
            worker_info=None,
            robot_info=None,
            env_idx=env_idx,
            env_cfg=self.cfg,
        )

    @property
    def offline_task(self) -> FR3OfflineTask:
        """Return the one local task without device discovery or placement."""
        return self.env.envs[0].unwrapped

    def step(
        self, actions: Sequence[Mapping[str, ActionRequest]], auto_reset: bool = False
    ) -> StepResult:
        """Return native batched tensors while requiring one typed action pair."""
        if auto_reset:
            self.offline_task.close()
            raise ValueError(
                "Offline FR3 requires explicit reset and mock authorization"
            )
        if (
            isinstance(actions, (np.ndarray, Mapping))
            or not isinstance(actions, (list, tuple))
            or len(actions) != 1
        ):
            self.offline_task.close()
            raise ActionRejected(
                "action_shape",
                "Expected a batch containing one typed normalized action pair",
            )
        raw, reward, terminated, truncated, infos = self.env.step(actions)
        self._elapsed_steps += 1
        truncated = self.elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            self.offline_task.robot.reset()
            self.offline_task._needs_reset = True
        infos = self._record_metrics(
            reward, terminated, np.zeros(1, dtype=bool), np.zeros(1, dtype=bool), infos
        )
        return (
            self._wrap_obs(raw),
            to_tensor(reward.astype(np.float32)),
            to_tensor(terminated),
            to_tensor(truncated),
            infos,
        )

    def close(self) -> None:
        """Release owned mock workers even after a previous close/reset cycle."""
        try:
            self.offline_task.close()
        finally:
            self.env.close()
