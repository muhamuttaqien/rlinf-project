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

"""Operator consent for bounded native FR3 evaluation."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from rlinf.robotics.parts.transports.ros2.fr3_runtime import supervised_duration_s
from rlinf.utils.logging import get_logger

NORM_SHA256 = "9bc91e36bd59e20e57f073a578a5c8e23977598a58f6ccd0f793c410522f9cba"


def validate_fr3_contract(cfg: Any) -> None:
    """Check the trained request architecture and a single evaluation episode."""
    duration = supervised_duration_s(
        OmegaConf.select(cfg, "runner.fr3_request_duration_s", default=3)
    )
    required = {
        "runner.only_eval": True,
        "rollout.model.model_type": "openpi",
        "rollout.model.action_dim": 7,
        "rollout.model.num_action_chunks": 10,
        "rollout.model.rewrite_chunk_gripper": False,
        "rollout.model.openpi.task": "eval",
        "rollout.model.openpi.config_name": "pi05_fr3_peg_request_offline",
        "rollout.model.openpi.action_horizon": 10,
        "rollout.model.openpi.action_chunk": 10,
        "rollout.model.openpi.model_action_dim": 32,
        "rollout.model.openpi.discrete_state_input": False,
        "rollout.pipeline_stage_num": 1,
        "env.eval.total_num_envs": 1,
        "env.eval.auto_reset": False,
        "env.eval.rollout_epoch": 1,
        "env.eval.max_steps_per_rollout_epoch": int(duration * 10),
        "env.eval.max_episode_steps": int(duration * 10),
        "env.eval.main_image_key": "base",
        "env.eval.teleop": "none",
        "env.eval.override_cfg.manual_episode_control_only": True,
    }
    for key, expected in required.items():
        value = OmegaConf.select(cfg, key)
        if value != expected:
            raise ValueError(
                f"FR3 request evaluation requires {key}={expected!r}, found {value!r}"
            )
    if (
        cfg.env.get("train")
        or cfg.get("actor")
        or OmegaConf.select(cfg, "runner.rtc.enabled", default=False)
    ):
        raise ValueError(
            "FR3 request evaluation requires evaluation-only workers with RTC disabled"
        )
    stats = Path(cfg.rollout.model.openpi_data.norm_stats_path)
    if hashlib.sha256(stats.read_bytes()).hexdigest() != NORM_SHA256:
        raise ValueError(
            "Normalization statistics differ from the request SFT training artifact"
        )
    checkpoint = Path(cfg.runner.ckpt_path)
    if (
        checkpoint.name != "full_weights.pt"
        or "global_step_3000" not in checkpoint.parts
        or not checkpoint.is_file()
        or checkpoint.stat().st_size < 1_000_000_000
    ):
        raise ValueError(
            "Expected the existing global_step_3000/full_weights.pt SFT artifact"
        )


class FR3OperatorSession:
    """One local consent lifetime, shared by driver and passive ROS peer."""

    def __init__(self, duration_s: float = 3.0) -> None:
        self.duration_s = supervised_duration_s(duration_s)
        self.directory = Path(tempfile.mkdtemp(prefix="rlinf-fr3-native-"))
        (self.directory / "owner.json").write_text(
            json.dumps({"pid": os.getpid(), "duration_s": self.duration_s})
        )
        atexit.register(self.stop)
        get_logger().info(
            "FR3 evidence and operator STOP file directory: %s", self.directory
        )

    def confirm(self) -> None:
        """Require a present operator after models and read-only state are ready."""
        if not sys.stdin.isatty():
            raise RuntimeError(
                "FR3 evaluation requires an interactive operator terminal"
            )
        confirmation = f"RUN FR3 {self.duration_s:g}S"
        get_logger().warning(
            "Existing NUC controller and bridge only. Confirm a clear workspace, "
            "a designated operator at the emergency stop, and exclusive policy ownership. "
            "Ctrl-C or touching %s/STOP requests neutral; neutral holds the NUC joint target. "
            "Use the hardware emergency stop for immediate robot stop. "
            "Type %s to permit one %g-second request window.",
            self.directory,
            confirmation,
            self.duration_s,
        )
        if input("FR3 operator confirmation: ").strip() != confirmation:
            raise RuntimeError(
                f"Operator did not confirm the {self.duration_s:g}-second evaluation"
            )
        (self.directory / "ARMED").write_text(confirmation + "\n")

    def stop(self) -> None:
        """Latch neutral without making a ROS call or modifying the controller."""
        (self.directory / "STOP").touch()
        (self.directory / "ARMED").unlink(missing_ok=True)


def prepare_fr3_session(cfg: Any) -> FR3OperatorSession | None:
    """Derive an isolated session only for the FR3 request Gymnasium task."""
    if OmegaConf.select(cfg, "env.eval.init_params.id") != "FR3RequestEnv-v1":
        return None
    validate_fr3_contract(cfg)
    if "session_dir" in cfg.env.eval.override_cfg:
        raise ValueError("session_dir is derived by the evaluation driver")
    session = FR3OperatorSession(
        OmegaConf.select(cfg, "runner.fr3_request_duration_s", default=3)
    )
    OmegaConf.update(
        cfg, "env.eval.override_cfg.session_dir", str(session.directory), force_add=True
    )
    return session
