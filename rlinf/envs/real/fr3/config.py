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

"""Strict OmegaConf configuration for synthetic FR3 integration only."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from rlinf.robotics.parts.ros2_observation import (
    FR3ROS2Observation,
    RobotiqROS2Observation,
)
from rlinf.robotics.parts.transports.ros2.action_contract import FR3_KIND, ROBOTIQ_KIND
from rlinf.robotics.parts.transports.ros2.protocol import FR3_JOINTS

ENV_ID = "FR3OfflineCollect-v0"
ENV_TYPE = "fr3_offline"
CAMERAS = {"d405": ("D405", "323743071035"), "d435i": ("D435i", "327743060094")}
STATE_FIELDS = tuple(
    sorted((*FR3ROS2Observation.STATE_FIELDS, *RobotiqROS2Observation.STATE_FIELDS))
)


@dataclass
class CameraFixtureConfig:
    """Audited camera identity and separate synthetic image dimensions."""

    model: str = ""
    serial_number: str = ""
    fixture_resolution: list[int] = field(default_factory=lambda: [32, 24])
    physical_role: str | None = None
    physical_resolution: list[int] | None = None
    calibration: str | None = None


@dataclass
class SpaceMouseFixtureConfig:
    """Audited USB identity with unresolved physical selection and mapping."""

    device: str = "SpaceMouse Wireless BT"
    usb_vendor_id: str = "256f"
    usb_product_id: str = "c63a"
    serial_number: str | None = None
    device_index: int | None = None
    controller_mapping: str | None = None


@dataclass
class SyntheticProfileConfig:
    """Mock-only bounds and timing; these never qualify physical hardware."""

    profile_id: str = "synthetic:fr3-config-v1"
    approval_id: str = "synthetic:fixture-only"
    calibration_id: str = "synthetic:not-physical-calibration"
    scope: str = "offline_synthetic"
    arm_lower: list[float] = field(default_factory=lambda: [-1.0] * 6)
    arm_upper: list[float] = field(default_factory=lambda: [1.0] * 6)
    gripper_lower: list[float] = field(default_factory=lambda: [0.0])
    gripper_upper: list[float] = field(default_factory=lambda: [1.0])
    max_action_age_ns: int = 1_000_000_000
    max_lifetime_ns: int = 1_000_000_000
    max_observation_age_ns: int = 1_000_000_000
    max_observation_skew_ns: int = 100_000_000
    max_timing_uncertainty_ns: int = 1_000_000
    acknowledgement_timeout_ns: int = 1_000_000_000
    result_timeout_ns: int = 1_000_000_000


@dataclass
class FR3OfflineConfig:
    """Fixed synthetic components and unresolved physical dependencies."""

    execution_mode: str = "offline_synthetic"
    robot_model: str = "FR3"
    gripper_model: str = "Robotiq 2F-85"
    joint_names: list[str] = field(default_factory=lambda: list(FR3_JOINTS))
    observation_transport: str = "synthetic_ros2"
    arm_observation_adapter: str = "FR3ROS2Observation"
    gripper_observation_adapter: str = "RobotiqROS2Observation"
    arm_domain_id: int = 70
    gripper_domain_id: int = 71
    action_transport: str = "mock_ros2"
    arm_action_adapter: str = "FR3OfflineAction"
    gripper_action_adapter: str = "RobotiqOfflineAction"
    arm_action_kind: str = FR3_KIND
    gripper_action_kind: str = ROBOTIQ_KIND
    arm_action_dimension: int = 6
    gripper_action_dimension: int = 1
    observation_fields: list[str] = field(default_factory=lambda: list(STATE_FIELDS))
    source_id: str = "synthetic:fr3-config-validation"
    task_description: str = "synthetic FR3 configuration validation (unscored)"
    cameras: dict[str, CameraFixtureConfig] = field(default_factory=dict)
    spacemouse: SpaceMouseFixtureConfig = field(default_factory=SpaceMouseFixtureConfig)
    synthetic_profile: SyntheticProfileConfig = field(
        default_factory=SyntheticProfileConfig
    )
    physical_calibration: str | None = None
    reward_semantics: str = "unscored_fixture"
    is_dummy: bool = True


def validate_task_config(values: Any) -> FR3OfflineConfig:
    """Resolve supported fields and refuse every live-backend or mapping choice."""
    if not values or values.get("synthetic_profile") is None:
        raise ValueError("Expected an explicit synthetic_profile configuration")
    for key in ("profile_id", "approval_id", "calibration_id", "scope"):
        if key not in values["synthetic_profile"]:
            raise ValueError(f"Expected explicit synthetic_profile.{key}")
    try:
        merged = OmegaConf.merge(OmegaConf.structured(FR3OfflineConfig), values)
        OmegaConf.resolve(merged)
        config = OmegaConf.to_object(merged)
    except OmegaConfBaseException as exc:
        raise ValueError(f"Invalid offline task configuration: {exc}") from exc
    expected = FR3OfflineConfig()
    fixed = (
        "execution_mode",
        "robot_model",
        "gripper_model",
        "joint_names",
        "observation_transport",
        "arm_observation_adapter",
        "gripper_observation_adapter",
        "arm_domain_id",
        "gripper_domain_id",
        "action_transport",
        "arm_action_adapter",
        "gripper_action_adapter",
        "arm_action_kind",
        "gripper_action_kind",
        "arm_action_dimension",
        "gripper_action_dimension",
        "observation_fields",
        "reward_semantics",
        "is_dummy",
    )
    for name in fixed:
        if getattr(config, name) != getattr(expected, name):
            raise ValueError(
                f"Expected offline FR3 {name}={getattr(expected, name)!r}; got {getattr(config, name)!r}"
            )
    if config.physical_calibration is not None:
        raise ValueError("Physical calibration cannot qualify this offline environment")
    if not config.source_id.startswith(
        "synthetic:"
    ) or not config.source_id.removeprefix("synthetic:"):
        raise ValueError("Expected an explicit synthetic source identifier")
    profile = config.synthetic_profile
    if profile.scope != "offline_synthetic" or any(
        not getattr(profile, key).startswith("synthetic:")
        for key in ("profile_id", "approval_id", "calibration_id")
    ):
        raise ValueError(
            "Expected explicitly synthetic profile, approval and calibration identifiers"
        )
    if set(config.cameras) != set(CAMERAS):
        raise ValueError("Expected both d405 and d435i camera fixtures")
    for name, camera in config.cameras.items():
        if (camera.model, camera.serial_number) != CAMERAS[name]:
            raise ValueError(f"Expected audited {name} model/serial {CAMERAS[name]}")
        if len(camera.fixture_resolution) != 2 or any(
            type(v) is not int or not 1 <= v <= 1024 for v in camera.fixture_resolution
        ):
            raise ValueError("Expected fixture width/height integers in [1,1024]")
        if any(
            getattr(camera, field) is not None
            for field in ("physical_role", "physical_resolution", "calibration")
        ):
            raise ValueError(
                "Physical camera roles, profiles and calibration remain unresolved"
            )
    if (
        config.cameras["d405"].fixture_resolution
        != config.cameras["d435i"].fixture_resolution
    ):
        raise ValueError(
            "Expected equal synthetic image dimensions for native view stacking"
        )
    if config.spacemouse != SpaceMouseFixtureConfig():
        raise ValueError(
            "Expected audited SpaceMouse identity with unresolved device selection/mapping"
        )
    mock_profiles(config)
    return config


def validate_environment_config(cfg: DictConfig) -> FR3OfflineConfig:
    """Check native env settings before construction, with no connection or worker."""
    if (
        cfg.get("env_type") != ENV_TYPE
        or cfg.get("init_params", {}).get("id") != ENV_ID
    ):
        raise ValueError(f"Expected env_type={ENV_TYPE} and init_params.id={ENV_ID}")
    fixed = {
        "total_num_envs": 1,
        "group_size": 1,
        "auto_reset": False,
        "ignore_terminations": False,
        "use_fixed_reset_state_ids": False,
        "teleop": "none",
        "reward_mode": "raw",
        "wrap_obs_mode": "simple",
        "main_image_key": "d435i",
    }
    for key, value in fixed.items():
        if cfg.get(key) != value:
            raise ValueError(
                f"Expected offline env {key}={value!r}; got {cfg.get(key)!r}"
            )
    allowed = set(fixed) | {
        "env_type",
        "seed",
        "max_episode_steps",
        "max_steps_per_rollout_epoch",
        "init_params",
        "override_cfg",
        "video_cfg",
        "data_collection",
    }
    if set(cfg) - allowed:
        raise ValueError(
            f"Unsupported offline environment settings: {sorted(set(cfg) - allowed)}"
        )
    if (
        set(cfg.init_params) != {"id", "num_envs"}
        or cfg.init_params.num_envs is not None
    ):
        raise ValueError("Expected native init_params id and null num_envs")
    if (
        type(cfg.seed) is not int
        or cfg.seed < 0
        or type(cfg.max_episode_steps) is not int
        or cfg.max_episode_steps < 1
    ):
        raise ValueError(
            "Expected nonnegative seed and positive synthetic episode horizon"
        )
    if (
        cfg.video_cfg.get("save_video") is not False
        or cfg.data_collection.get("enabled") is not False
    ):
        raise ValueError(
            "Offline configuration validation cannot enable recording or video output"
        )
    if cfg.data_collection.get("export_format") not in ("pickle", "lerobot"):
        raise ValueError("Expected native future export_format pickle or lerobot")
    if set(cfg.video_cfg) != {"save_video", "info_on_video", "video_base_dir"} or set(
        cfg.data_collection
    ) != {"enabled", "save_dir", "export_format", "only_success", "robot_type", "fps"}:
        raise ValueError(
            "Expected only supported native recording configuration fields"
        )
    if (
        cfg.data_collection.robot_type != "fr3"
        or type(cfg.data_collection.fps) is not int
        or cfg.data_collection.fps <= 0
    ):
        raise ValueError("Expected fr3 recording metadata with a positive fixture fps")
    return validate_task_config(cfg.override_cfg)


def validation_report(cfg: DictConfig) -> dict[str, Any]:
    """Describe registration inputs and unresolved physical dependencies only."""
    task = validate_environment_config(cfg.env.eval)
    if cfg.runner.num_data_episodes < 1:
        raise ValueError("Expected a positive episode-count setting")
    return {
        "execution_mode": task.execution_mode,
        "synthetic": True,
        "env_type": ENV_TYPE,
        "gym_id": ENV_ID,
        "robot": task.robot_model,
        "state_fields": list(STATE_FIELDS),
        "actions": {
            "arm": {"kind": task.arm_action_kind, "dimension": 6},
            "end_effector": {"kind": task.gripper_action_kind, "dimension": 1},
        },
        "observation_domains": {"arm": 70, "gripper": 71},
        "action_ros_domains": None,
        "camera_fixture_resolution": {
            k: v.fixture_resolution for k, v in task.cameras.items()
        },
        "unresolved": [
            "physical camera roles/profiles/calibration",
            "SpaceMouse selection and controller mapping",
            "FR3 frame/TCP and gripper calibration",
            "timing, ownership, safeguards and operator qualification",
            "native recorder integration",
        ],
        "qualification_gate": "NO_GO",
        "physical_command_authorization": "NOT_GRANTED",
        "physical_safety_qualification": "NOT_QUALIFIED",
    }


def mock_profiles(config: FR3OfflineConfig) -> dict[str, Any]:
    """Build and validate synthetic approvals without opening mock workers."""
    from rlinf.robotics.parts.transports.ros2.action_contract import (
        InterfaceIdentity,
        MockAuthorization,
        MockLimitProfile,
        validate_approval,
    )

    settings = config.synthetic_profile
    result = {}
    for name, kind, lower, upper in (
        ("arm", FR3_KIND, settings.arm_lower, settings.arm_upper),
        ("end_effector", ROBOTIQ_KIND, settings.gripper_lower, settings.gripper_upper),
    ):
        interface = InterfaceIdentity.mock(
            kind, "mock:" + name + "-config-v1", "synthetic:fr3-config-v1"
        )
        profile = MockLimitProfile(
            settings.profile_id + ":" + name,
            settings.approval_id,
            settings.calibration_id,
            interface,
            tuple(lower),
            tuple(upper),
            settings.max_action_age_ns,
            settings.max_lifetime_ns,
            settings.max_observation_age_ns,
            settings.max_observation_skew_ns,
            settings.max_timing_uncertainty_ns,
            settings.acknowledgement_timeout_ns,
            settings.result_timeout_ns,
            settings.scope,
        )
        auth = MockAuthorization(
            "synthetic:validation",
            config.source_id,
            "validation-session",
            "validation-authority",
            profile.profile_id,
            "validation-clock",
            60_000_000_000,
        )
        validate_approval(
            profile,
            auth,
            interface,
            auth.session_id,
            auth.authority_id,
            auth.clock_domain,
            0,
        )
        result[name] = profile
    return result
