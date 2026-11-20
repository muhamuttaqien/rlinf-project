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

"""Hardware-inaccessible FR3 collection using native trajectory/replay APIs."""

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.schema.embodied_trajectory import TrajectoryAccumulator
from rlinf.data.schema.embodied_types import TrajectoryStep
from rlinf.data.storage.fr3_synthetic import (
    GATE,
    PROVENANCE_SCHEMA,
    SCHEMA,
    _validate_dataset,
    append_record,
    file_digest,
    json_value,
    observation_record,
    read_json,
    write_json,
)
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.envs import get_env_cls
from rlinf.envs.real.fr3.config import ENV_ID, STATE_FIELDS, validate_environment_config
from rlinf.envs.real.fr3.offline import FR3OfflineRealWorldEnv
from rlinf.robotics.parts.fr3_fixtures import (
    STREAMS,
    SyntheticFrame,
    validate_alignment,
)
from rlinf.robotics.parts.transports.ros2.action_contract import (
    FR3_KIND,
    ROBOTIQ_KIND,
    ManualClock,
)


@dataclass(frozen=True)
class SyntheticScenario:
    """Fixed offline failure fixtures; none can select a device or live endpoint."""

    name: str = "normal"
    at_step: int = 0

    def __post_init__(self) -> None:
        if self.name not in {
            "normal",
            "invalid_action",
            "expired_action",
            "no_authorization",
            "mock_fail",
            "mock_reject",
            "ack_timeout",
            "worker_crash",
            "ipc_failure",
            "missing_camera",
            "delayed_camera",
            "stale_observation",
            "sample_skew",
            "interrupted",
        }:
            raise ValueError("Expected a fixed synthetic failure scenario")
        if type(self.at_step) is not int or self.at_step < 0:
            raise ValueError("Expected a nonnegative scenario step")


def validate_collection_config(cfg: DictConfig) -> dict:
    """Validate the fixed synthetic environment and bounded recorder resources."""
    task = validate_environment_config(cfg.env.eval)
    if cfg.cluster.num_nodes != 1 or cfg.cluster.get("node_groups"):
        raise ValueError(
            "Expected local CPU collection without physical hardware allocation"
        )
    if (
        type(cfg.runner.num_data_episodes) is not int
        or not 1 <= cfg.runner.num_data_episodes <= 100
    ):
        raise ValueError("Expected 1..100 synthetic episodes")
    settings = OmegaConf.to_container(cfg.offline_collection, resolve=True)
    expected_keys = {
        "schema_version",
        "step_period_ns",
        "max_sample_age_ns",
        "max_sample_skew_ns",
    }
    if (
        set(settings) != expected_keys
        or type(settings["schema_version"]) is not int
        or settings["schema_version"] != 1
    ):
        raise ValueError(
            "Expected synthetic recorder schema 1 and supported timing fields"
        )
    for name in expected_keys - {"schema_version"}:
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError("Expected positive integer synthetic timing settings")
    if settings["step_period_ns"] < 10_000_000 or cfg.env.eval.max_episode_steps > 1000:
        raise ValueError("Expected fixture period >=10ms and <=1000 steps per episode")
    image_bytes = sum(
        c.fixture_resolution[0] * c.fixture_resolution[1] * 3
        for c in task.cameras.values()
    )
    if (
        image_bytes * 2 * cfg.env.eval.max_episode_steps * cfg.runner.num_data_episodes
        > 128 * 1024**2
    ):
        raise ValueError("Synthetic dataset exceeds the 128MiB image budget")
    if get_env_cls(cfg.env.eval.env_type) is not FR3OfflineRealWorldEnv:
        raise ValueError("Expected only the fixed fr3_offline environment factory")
    return settings


def _tensor_observation(observation: dict) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in observation.items()
        if key != "task_descriptions"
    }


def _error(exc: BaseException, root: Path) -> dict:
    reason = str(exc).replace(str(root), "<dataset>")
    return {
        "type": type(exc).__name__,
        "code": getattr(exc, "code", None),
        "reason": reason[:2048],
    }


def _episode(
    cfg: DictConfig,
    env: FR3OfflineRealWorldEnv,
    root: Path,
    number: int,
    clock: ManualClock,
    settings: dict,
    scenario: SyntheticScenario,
) -> tuple[Any, dict]:
    task = env.offline_task
    directory = root / "episodes" / f"episode_{number:06d}"
    directory.mkdir(parents=True)
    metadata = {
        "schema": PROVENANCE_SCHEMA,
        "episode_id": f"synthetic:episode:{number:06d}",
        "dataset_id": clock.domain.removeprefix("synthetic:"),
        "status": "WRITING",
        "step_count": 0,
        "start_logical_ns": clock.now_ns(),
        "start_recorder_ns": task.clock.now_ns(),
        "recorder_clock_domain": task.clock.domain,
        "reward_semantics": "unscored_fixture_zero",
        "reward_evaluated": False,
        "success_evaluated": False,
        "intervention_semantics": "no_human_or_policy_intervention",
        **GATE,
    }
    write_json(directory / "episode.json", metadata)
    accumulator = TrajectoryAccumulator(
        max_episode_length=cfg.env.eval.max_episode_steps
    )
    frame = SyntheticFrame(
        0, clock.now_ns(), clock.domain, seed=cfg.env.eval.seed + number
    )
    task.set_synthetic_frame(frame)
    last_events = {
        name: max((e["event_index"] for e in part.events()), default=0)
        for name, part in task.actions.items()
    }
    active_record = None
    try:
        observation, reset_info = env.reset(seed=cfg.env.eval.seed + number)
        metadata["environment_episode_id"] = reset_info["episode_id"][0]
        metadata["sessions"] = {
            name: {"session_id": part.session_id, "authority_id": part.authority_id}
            for name, part in task.actions.items()
        }
        for i in range(cfg.env.eval.max_episode_steps):
            timings = {name: frame.timing(name) for name in STREAMS}
            active_record = {
                "schema": PROVENANCE_SCHEMA,
                "event": "step_requested",
                "step_index": i,
                "recorder_ns": task.clock.now_ns(),
                "recorder_clock_domain": task.clock.domain,
                "logical_action_ns": clock.now_ns(),
                "synthetic_clock_domain": clock.domain,
                **GATE,
            }
            before = task.synthetic_observation()
            active_record["observation_before"] = observation_record(before)
            active_record["operator_request"] = {
                "source_id": task.config.source_id,
                "raw_input": json_value(before["spacemouse"]),
                "interpretation": "scripted_fixture_request_v1; physical device mapping unverified",
                "arm_kind": FR3_KIND,
                "gripper_kind": ROBOTIQ_KIND,
                "arm_values": list(frame.axes()),
                "normalized_closure": frame.closure(),
                "input_timing": frame.timing("spacemouse"),
                "physical_mapping_verified": False,
            }
            if scenario.name != "no_authorization" or i != scenario.at_step:
                task.authorize_synthetic()
            request = task.create_normalized_action(
                frame.axes(), frame.closure(), sequence=i + 1
            )
            if i == scenario.at_step:
                if scenario.name == "invalid_action":
                    request["arm"] = replace(request["arm"], values=(float("nan"),) * 6)
                elif scenario.name == "expired_action":
                    request["arm"] = replace(
                        request["arm"], deadline_ns=request["arm"].generated_ns
                    )
                elif scenario.name in {
                    "mock_fail",
                    "mock_reject",
                    "worker_crash",
                    "ipc_failure",
                }:
                    task.actions["arm"].configure_mock_behavior(
                        {"mock_fail": "fail", "mock_reject": "reject"}.get(
                            scenario.name, scenario.name
                        )
                    )
                elif scenario.name == "ack_timeout":
                    task.actions["end_effector"].configure_mock_behavior("drop")
            active_record["typed_requests"] = json_value(request)
            append_record(directory / "steps.jsonl", active_record)
            validate_alignment(
                timings, settings["max_sample_age_ns"], settings["max_sample_skew_ns"]
            )
            clock.advance(settings["step_period_ns"])
            next_frame = SyntheticFrame(
                i + 1, clock.now_ns(), clock.domain, seed=cfg.env.eval.seed + number
            )
            if i == scenario.at_step:
                if scenario.name == "missing_camera":
                    next_frame = replace(next_frame, missing=("d405",))
                elif scenario.name == "delayed_camera":
                    next_frame = replace(
                        next_frame,
                        arrival_offsets_ns={**next_frame.arrival_offsets_ns, "d405": 1},
                    )
                elif scenario.name == "stale_observation":
                    next_frame = replace(
                        next_frame, source_offsets_ns=dict.fromkeys(STREAMS, -200000000)
                    )
                elif scenario.name == "sample_skew":
                    next_frame = replace(
                        next_frame,
                        source_offsets_ns={
                            **next_frame.source_offsets_ns,
                            "d405": -40_000_000,
                        },
                    )
                elif scenario.name == "interrupted":
                    raise KeyboardInterrupt("Synthetic interrupted episode fixture")
            active_record["next_sample_timing"] = {
                name: next_frame.timing(name) for name in STREAMS
            }
            validate_alignment(
                active_record["next_sample_timing"],
                settings["max_sample_age_ns"],
                settings["max_sample_skew_ns"],
            )
            task.set_synthetic_frame(next_frame)
            next_observation, reward, terminated, truncated, _ = env.step([request])
            states = {
                name: part.get_state()["mock_action"]
                for name, part in task.actions.items()
            }
            if (
                states["arm"]["last_disposition"]
                not in ("acceptance_unknown", "mock_canceled")
                or states["end_effector"]["last_disposition"] != "mock_completed"
            ):
                raise ValueError(
                    "Mock submission/result failure: " + json.dumps(states)
                )
            # End this mock action interval explicitly; no physical stop/hold is implied.
            for part in task.actions.values():
                part.reset()
            events = {}
            for name, part in task.actions.items():
                new_events = [
                    e for e in part.events() if e["event_index"] > last_events[name]
                ]
                events[name] = [
                    e
                    for e in new_events
                    if e["request"] and e["request"]["sequence"] == i + 1
                ]
                last_events[name] = max(
                    (e["event_index"] for e in part.events()), default=last_events[name]
                )
            outcome = {
                "schema": PROVENANCE_SCHEMA,
                "event": "step_outcome",
                "step_index": i,
                "valid": True,
                "recorder_ns": task.clock.now_ns(),
                "recorder_clock_domain": task.clock.domain,
                "action_events": events,
                "mock_states": states,
                "physical_controller_acceptance": "UNAVAILABLE",
                "physical_grasp_success": None,
                "observation_after": observation_record(task.synthetic_observation()),
                **GATE,
            }
            append_record(directory / "steps.jsonl", outcome)
            dense_action = torch.tensor(
                [list(request["arm"].values) + list(request["end_effector"].values)],
                dtype=torch.float32,
            )
            accumulator.append(
                TrajectoryStep(
                    actions=dense_action,
                    rewards=reward.float().unsqueeze(1),
                    dones=(terminated | truncated).unsqueeze(1),
                    terminations=terminated.unsqueeze(1),
                    truncations=truncated.unsqueeze(1),
                    forward_inputs={"action": dense_action},
                    curr_obs=_tensor_observation(observation),
                    next_obs=_tensor_observation(next_observation),
                )
            )
            metadata["step_count"] += 1
            active_record = None
            observation, frame = next_observation, next_frame
        metadata.update(
            status="CAPTURED",
            end_logical_ns=clock.now_ns(),
            end_recorder_ns=task.clock.now_ns(),
            termination_reason="synthetic_horizon",
        )
        write_json(directory / "episode.json", metadata)
        return accumulator.to_trajectory(), {
            "metadata_path": str((directory / "episode.json").relative_to(root)),
            "provenance_path": str((directory / "steps.jsonl").relative_to(root)),
        }
    except BaseException as exc:
        if active_record is not None:
            append_record(
                directory / "steps.jsonl",
                {
                    "schema": PROVENANCE_SCHEMA,
                    "event": "step_outcome",
                    "step_index": active_record["step_index"],
                    "valid": False,
                    "error": _error(exc, root),
                    "next_sample_timing": active_record.get("next_sample_timing"),
                    "action_events": {n: p.events() for n, p in task.actions.items()},
                    **GATE,
                },
            )
        metadata.update(status="INCOMPLETE", error=_error(exc, root))
        write_json(directory / "episode.json", metadata)
        raise


def collect_synthetic_dataset(
    cfg: DictConfig, output: str | Path, *, scenario: SyntheticScenario | None = None
) -> dict:
    """Record and verify synthetic episodes; an existing output is never reused."""
    settings = validate_collection_config(cfg)
    scenario = scenario or SyntheticScenario()
    if scenario.at_step >= cfg.env.eval.max_episode_steps:
        raise ValueError("Expected scenario step within the configured episode")
    root = Path(output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    identity = "dataset:" + uuid.uuid4().hex
    clock = ManualClock(domain="synthetic:" + identity)
    config_json = OmegaConf.to_container(cfg, resolve=True)
    manifest = {
        "schema": SCHEMA,
        "provenance_schema": PROVENANCE_SCHEMA,
        "status": "WRITING",
        "dataset_id": identity,
        "environment_id": ENV_ID,
        "env_type": "fr3_offline",
        "config_sha256": hashlib.sha256(
            json.dumps(config_json, sort_keys=True).encode()
        ).hexdigest(),
        "recorder_software": {
            "version": "fr3-offline-recorder-v1",
            "source_sha256": file_digest(Path(__file__)),
        },
        "seed": cfg.env.eval.seed,
        "started_wall_ns": time.time_ns(),
        "synthetic_clock_domain": clock.domain,
        "observation_schema_version": 1,
        "action_schema_version": 1,
        "state_fields": list(STATE_FIELDS),
        "action_kinds": [FR3_KIND, ROBOTIQ_KIND],
        "dense_action_semantics": "submitted_mock_normalized_payload; never physical execution",
        "camera_roles": {"d405": None, "d435i": None},
        "camera_tensor_slots": {"d435i": "main_images", "d405": "extra_view_images[0]"},
        "alignment": {
            "policy": "latest_before_action_no_substitution",
            "max_age_ns": settings["max_sample_age_ns"],
            "max_skew_ns": settings["max_sample_skew_ns"],
        },
        "step_period_ns": settings["step_period_ns"],
        "expected_episodes": cfg.runner.num_data_episodes,
        "expected_steps_per_episode": cfg.env.eval.max_episode_steps,
        "robot_model": "FR3",
        "gripper_model": "Robotiq 2F-85",
        "camera_identities": {
            name: {"model": camera.model, "serial_number": camera.serial_number}
            for name, camera in cfg.env.eval.override_cfg.cameras.items()
        },
        "episodes": [],
        "errors": [],
        **GATE,
    }
    env = buffer = None
    try:
        write_json(root / "manifest.json", manifest)
        env = get_env_cls("fr3_offline")(cfg.env.eval)
        manifest["recorder_clock_domain"] = env.offline_task.clock.domain
        manifest["interfaces"] = {
            name: json_value(part.interface)
            for name, part in env.offline_task.actions.items()
        }
        buffer = TrajectoryReplayBuffer(
            seed=cfg.env.eval.seed,
            enable_cache=False,
            auto_save=True,
            auto_save_path=str(root / ".buffer"),
            sample_window_size=0,
        )
        for number in range(cfg.runner.num_data_episodes):
            trajectory, entry = _episode(
                cfg, env, root, number, clock, settings, scenario
            )
            buffer.add_trajectories([trajectory])
            manifest["episodes"].append(entry)
            write_json(root / "manifest.json", manifest)
            clock.advance(settings["step_period_ns"])
        buffer.save_checkpoint(str(root / "demos"))
        env.close()
        env = None
        buffer.close()
        buffer = None
        shutil.rmtree(root / ".buffer")
        for entry in manifest["episodes"]:
            path = root / entry["metadata_path"]
            episode = read_json(path)
            episode["status"] = "COMPLETE"
            write_json(path, episode)
        for path in root.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        manifest["files"] = {
            str(p.relative_to(root)): file_digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p != root / "manifest.json"
        }
        manifest["status"] = "VERIFYING"
        write_json(root / "manifest.json", manifest)
        report = _validate_dataset(root, "VERIFYING")
        manifest.update(
            status="COMPLETE", ended_wall_ns=time.time_ns(), validation=report
        )
        write_json(root / "manifest.json", manifest)
        return report
    except BaseException as exc:
        manifest["status"] = "INCOMPLETE"
        manifest["errors"].append(_error(exc, root))
        for resource in (env, buffer):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as cleanup:
                    manifest["errors"].append(_error(cleanup, root))
        for entry in manifest["episodes"]:
            try:
                path = root / entry["metadata_path"]
                episode = read_json(path)
                episode.update(status="INCOMPLETE", dataset_finalized=False)
                write_json(path, episode)
            except (OSError, ValueError, KeyError) as incomplete_status:
                manifest["errors"].append(_error(incomplete_status, root))
        try:
            write_json(root / "manifest.json", manifest)
        except OSError:
            pass  # A disk failure can leave WRITING/VERIFYING, which loaders reject.
        raise
