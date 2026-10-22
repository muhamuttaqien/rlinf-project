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

"""One live preflight of clean-v1 StackCube scene, controller and SFT contract.

Runs on the workstation before collecting demonstrations. No model/GPU SFT
weights are loaded; no dataset is written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from toolkits.lerobot.stackcube_ee_delta import (
    ACTION_DIM, ENV_ID, STATE_DIM, extract_ee_observation, make_env,
)

CONFIG = Path(
    "examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_clean_v1.yaml"
)


def _single_number(value) -> float:
    data = np.asarray(value, dtype=np.float64)
    if data.size != 1:
        raise ValueError(f"Expected scalar controller limit, got {value!r}")
    return float(data.item())


def verify_clean_config() -> None:
    cfg = OmegaConf.load(CONFIG)
    model = cfg.actor.model
    if not (
        cfg.runner.logger.experiment_name == "maniskill_stackcube_pi05_clean_v1"
        and model.openpi.config_name
        == "pi05_stackcube_panda_easy_ee_delta_two_cam_state"
        and model.openpi.train_image_augmentation is False
        and model.openpi.action_horizon == 10
        and model.openpi.num_images_in_input == 2
        and model.openpi.discrete_state_input is True
        and model.action_dim == ACTION_DIM
        and cfg.runner.max_steps == 600
    ):
        raise AssertionError("Clean experiment SFT config is inconsistent")
    # The old dataset/weights must never appear in the new experiment's data
    # or logger identity. The pretrained checkpoint is resolved separately.
    if "stackcube_pi05_clean_v1" not in str(cfg.data.train_data_paths):
        raise AssertionError("SFT data path points away from clean-v1 dataset")


def verify_environment() -> None:
    env = make_env()
    try:
        base = env.unwrapped
        if str(base.control_mode) != "pd_ee_delta_pose":
            raise AssertionError("Wrong ManiSkill control mode")
        if int(np.prod(env.single_action_space.shape)) != ACTION_DIM:
            raise AssertionError(f"Not a 7D controller: {env.single_action_space}")
        controllers = base.agent._controller_configs
        arm = controllers["pd_ee_delta_pose"]["arm"]
        for key, expected in (
            ("pos_lower", -0.1), ("pos_upper", +0.1),
            ("rot_lower", -0.1), ("rot_upper", +0.1),
        ):
            actual = _single_number(getattr(arm, key))
            if not np.isclose(actual, expected, atol=1e-6):
                raise AssertionError(
                    f"Installed ManiSkill controller {key}={actual}, "
                    f"expected {expected}"
                )
        if not (
            arm.use_delta is True
            and arm.use_target is False
            and arm.normalize_action is True
            and arm.frame == "root_translation:root_aligned_body_rotation"
        ):
            raise AssertionError(
                "Installed Panda controller has unexpected delta/frame settings"
            )
        # The environment itself is the authority for seed reproducibility;
        # do not assume a particular RNG implementation.
        obs, _ = env.reset(seed=123)
        parsed = extract_ee_observation(env, obs)
        if parsed["state"].shape != (STATE_DIM,):
            raise AssertionError("Invalid live 8D EE state")
        if parsed["image"].shape != (224, 224, 3):
            raise AssertionError("Missing base RGB camera")
        if parsed["wrist_image"].shape != (224, 224, 3):
            raise AssertionError("Missing wrist RGB camera")
        if np.array_equal(parsed["image"], parsed["wrist_image"]):
            raise AssertionError("Two camera images are identical")
        first = np.concatenate(
            [
                np.asarray(base.cubeA.pose.p.detach().cpu()).ravel(),
                np.asarray(base.cubeB.pose.p.detach().cpu()).ravel(),
            ]
        )
        env.reset(seed=456)
        env.reset(seed=123)
        repeated = np.concatenate(
            [
                np.asarray(base.cubeA.pose.p.detach().cpu()).ravel(),
                np.asarray(base.cubeB.pose.p.detach().cpu()).ravel(),
            ]
        )
        np.testing.assert_allclose(first, repeated, rtol=0, atol=1e-6)
        print(
            f"ENVIRONMENT PASSED id={ENV_ID} action_dim={ACTION_DIM} "
            f"state_dim={STATE_DIM} cameras=2 controller=normalized_ee_delta"
        )
    finally:
        env.close()


def main() -> None:
    verify_clean_config()
    verify_environment()
    print("CLEAN PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
