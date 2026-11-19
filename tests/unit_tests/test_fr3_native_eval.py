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

"""Native FR3 request evaluation contracts against fake ROS and camera SDKs."""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.robotics.parts.transports.ros2.fr3_runtime import (
    ARM_TOPIC,
    GRIP_TOPIC,
    RequestWindow,
    request_vector,
)

ROOT = Path(__file__).resolve().parents[2]


def configuration(monkeypatch, name="realworld_fr3_pi05_request_eval"):
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(ROOT / "evaluations/realworld"), version_base="1.1"
    ):
        return compose(config_name=name)


@pytest.fixture
def fake_session(tmp_path, monkeypatch):
    """Launch the real ROS peer with only its ROS SDK boundary replaced."""
    from robot_mocks import ros2

    setup = tmp_path / "setup.bash"
    setup.write_text(":\n")
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text(
        "from robot_mocks.ros2 import install; install()\n"
    )
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(bootstrap), str(ROOT / "tests"), str(ROOT)))
    )
    monkeypatch.setenv("FR3_FAKE_WIRE_LOG", str(tmp_path / "wire.jsonl"))
    monkeypatch.setitem(sys.modules, "pyrealsense2", ros2.realsense())
    (tmp_path / "owner.json").write_text(json.dumps({"pid": os.getpid()}))
    return {
        "session_dir": str(tmp_path),
        "ros_python": sys.executable,
        "ros_setup": str(setup),
        "ros_overlay": str(setup),
        "manual_episode_control_only": True,
    }


@pytest.mark.parametrize(
    "bad", [[0] * 6, [float("nan")] * 7, [1.01] * 7, [0] * 6 + [1.1]]
)
def test_invalid_request_refuses_whole_sample(bad):
    with pytest.raises(ValueError):
        request_vector(bad)


def test_request_expiry_deadline_and_refresh_gap():
    window = RequestWindow()
    value = [0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.5]
    window.submit(value, 0, 0)
    for i in range(10):
        assert window.tick(i / 100, True) == value[:6]
    assert window.tick(0.1, True) == [0] * 6
    for i in range(11, 301):
        window.tick(i / 100, True)
    assert window.reason == "three_second_deadline"
    with pytest.raises(RuntimeError):
        window.submit(value, 3.01, 3.01)
    gap = RequestWindow()
    gap.submit(value, 0, 0)
    gap.tick(0, True)
    assert gap.tick(0.051, True) == [0] * 6
    assert gap.reason == "command_refresh_gap"


def test_neutral_gripper_debounce_and_health_stop():
    window = RequestWindow()
    for i in range(3):
        assert window.submit([0] * 6 + [0.5], i / 10, i / 10) is None
    assert [window.submit([0] * 6 + [1], t, t) for t in (0.3, 0.4, 0.5)] == [
        None,
        None,
        1,
    ]
    assert window.tick(0.5, False) == [0] * 6
    assert window.reason == "state_or_ownership_lost"
    with pytest.raises(RuntimeError):
        window.submit([0] * 7, 0.6, 0.6)


@pytest.mark.parametrize("bad", [0, -5, 4, 6, True, "5", float("nan"), float("inf")])
def test_unreviewed_duration_refused(bad):
    with pytest.raises(ValueError, match="3 or 5"):
        RequestWindow(bad)


def test_five_second_window_retains_expiry_and_fixed_deadline():
    window = RequestWindow(5)
    request = [0.4] * 6 + [0.5]
    for tick in range(500):
        now = tick / 100
        if tick % 10 == 0:
            window.submit(request, now, now)
        assert window.tick(now, True) == request[:6]
    assert window.deadline == 5
    assert window.tick(5, True) == [0] * 6
    assert window.reason == "five_second_deadline"
    with pytest.raises(RuntimeError, match="stopped"):
        window.submit(request, 5.01, 5.01)
    expired = RequestWindow(5)
    expired.submit(request, 0, 0)
    assert expired.tick(0.1, True) == [0] * 6
    assert expired.reason is None


def test_five_second_config_changes_only_window_and_step_limits(monkeypatch):
    from evaluations.fr3_operator import validate_fr3_contract

    base = configuration(monkeypatch)
    longer = configuration(monkeypatch, "realworld_fr3_pi05_request_eval_5s")
    for cfg, seconds in ((base, 3), (longer, 5)):
        assert cfg.runner.fr3_request_duration_s == seconds
        assert cfg.env.eval.max_steps_per_rollout_epoch == seconds * 10
        assert cfg.env.eval.max_episode_steps == seconds * 10
    normalized = OmegaConf.to_container(longer, resolve=True)
    normalized["runner"]["fr3_request_duration_s"] = 3
    normalized["env"]["eval"]["max_steps_per_rollout_epoch"] = 30
    normalized["env"]["eval"]["max_episode_steps"] = 30
    assert normalized == OmegaConf.to_container(base, resolve=True)
    longer.env.eval.max_episode_steps = 30
    with pytest.raises(ValueError, match="max_episode_steps"):
        validate_fr3_contract(longer)


def test_five_second_operator_confirmation_and_peer_metadata(tmp_path, monkeypatch):
    import io

    from evaluations.fr3_operator import FR3OperatorSession

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("tempfile.mkdtemp", lambda **kwargs: str(tmp_path))
    cfg = configuration(monkeypatch, "realworld_fr3_pi05_request_eval_5s")
    session = FR3OperatorSession(cfg.runner.fr3_request_duration_s)
    assert json.loads((tmp_path / "owner.json").read_text())["duration_s"] == 5
    monkeypatch.setattr(sys, "stdin", Terminal("RUN FR3 3S\n"))
    with pytest.raises(RuntimeError, match="5-second"):
        session.confirm()
    assert not (tmp_path / "ARMED").exists()
    monkeypatch.setattr(sys, "stdin", Terminal("RUN FR3 5S\n"))
    session.confirm()
    assert (tmp_path / "ARMED").read_text() == "RUN FR3 5S\n"
    session.stop()
    assert (tmp_path / "STOP").exists() and not (tmp_path / "ARMED").exists()


@pytest.fixture
def peer(tmp_path, monkeypatch, request):
    from robot_mocks import ros2

    saved = dict(sys.modules)
    ros2.install()
    monkeypatch.syspath_prepend(str(ROOT / "rlinf/robotics/parts/transports/ros2"))
    module = importlib.import_module("fr3_node")
    (tmp_path / "owner.json").write_text(
        json.dumps({"pid": os.getpid(), "duration_s": getattr(request, "param", 3)})
    )
    value = module.RosPeer(tmp_path, tmp_path / "evidence.jsonl")
    for _ in range(4):
        value.spin()
    assert value.healthy()
    yield value
    value.close()
    for name in set(sys.modules) - set(saved):
        if name.startswith(
            (
                "rclpy",
                "franka_msgs",
                "controller_manager_msgs",
                "geometry_msgs",
                "sensor_msgs",
                "std_msgs",
                "rcl_interfaces",
                "fr3_node",
                "fr3_runtime",
            )
        ):
            sys.modules.pop(name, None)
    for name, original in saved.items():
        if name.startswith(
            (
                "rclpy",
                "franka_msgs",
                "controller_manager_msgs",
                "geometry_msgs",
                "sensor_msgs",
                "std_msgs",
                "rcl_interfaces",
            )
        ):
            sys.modules[name] = original


def test_physical_joint_filter_and_state_order(peer):
    from robot_mocks.ros2 import message

    state = peer.observation()["state"]
    assert state == pytest.approx(
        [0.4, 0, 0.4, 0, 0, 0, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 3, 0.7859, 5]
    )
    before = peer.cache["gripper"]
    wrong = message("/joint_states", 71)
    wrong.name, wrong.position = ["robotiq_normalized_joint"], [0.8941]
    peer.receive("gripper", wrong)
    assert peer.cache["gripper"] is before
    peer.receive("gripper", before[0])
    assert peer.cache["gripper"] is before
    stale = (before[0], time.monotonic() - 0.26, before[2])
    peer.cache["gripper"] = stale
    with pytest.raises(RuntimeError, match="stale"):
        peer.observation()


@pytest.mark.parametrize(
    "failure", ["competitor", "controller", "frame", "fault", "STOP"]
)
def test_ros_runtime_failures_refuse_commands(peer, failure):
    (peer.session / "ARMED").write_text("fake hardware test\n")
    if failure == "competitor":
        peer.nodes[0].extra_publishers = 1
        peer.next_graph = 0
        peer.spin()
    elif failure == "controller":
        peer.controller_ok = False
    elif failure == "frame":
        peer.cache["pose"][0].header.frame_id = "wrong"
    elif failure == "fault":
        peer.cache["robot"][0].current_errors.fault = True
    else:
        (peer.session / "STOP").touch()
    with pytest.raises(RuntimeError):
        peer.command({"op": "request", "value": [0] * 7, "issued": time.monotonic()})
    assert peer.window.reason
    assert peer.pubs is None


def test_ros_reservation_and_operator_confirmation(peer):
    assert peer.pubs is None
    peer.command({"op": "reserve"})
    peer.spin()
    assert all(not values for values in peer.nodes[0].publishers.values())
    with pytest.raises(RuntimeError, match="confirmation"):
        peer.command(
            {"op": "request", "value": [0.3] * 6 + [0.5], "issued": time.monotonic()}
        )
    assert all(not values for values in peer.nodes[0].publishers.values())


@pytest.mark.parametrize("duration_s", [3, 5])
def test_native_env_chunks_with_real_subprocess(fake_session, monkeypatch, duration_s):
    """Exercise native vector/chunk wrapper, robot composition and ROS process."""
    from rlinf.envs.action_utils import prepare_actions
    from rlinf.envs.real.env import RealWorldEnv
    from rlinf.models.embodiment.openpi.policies.fr3_peg_request_policy import (
        Fr3PegRequestInputs,
    )
    from rlinf.models.embodiment.openpi.transforms.env import repack_env_obs

    monkeypatch.setenv("RLINF_ROS_SETUP", "preserve")
    name = "realworld_fr3_pi05_request_eval" + ("_5s" if duration_s == 5 else "")
    cfg = configuration(monkeypatch, name).env.eval
    cfg.override_cfg = OmegaConf.create(fake_session)
    directory = Path(fake_session["session_dir"])
    (directory / "owner.json").write_text(
        json.dumps({"pid": os.getpid(), "duration_s": duration_s})
    )
    env = RealWorldEnv(cfg, 1, 0, 1, None)
    try:
        obs, _ = env.reset()
        assert obs["states"].shape == (1, 19)
        assert obs["main_images"].shape == (1, 480, 640, 3)
        assert obs["extra_view_images"].shape == (1, 1, 480, 848, 3)
        assert obs["main_images"][0, 0, 0].tolist() == [33, 22, 11]
        assert not (directory / "wire.jsonl").exists()
        repacked = repack_env_obs(
            "pi05_fr3_peg_request_offline", obs, select_state=lambda x: x
        )
        per_sample = {
            key: (value[0].numpy() if isinstance(value, torch.Tensor) else value[0])
            for key, value in repacked.items()
        }
        converted = Fr3PegRequestInputs()(per_sample)
        assert converted["state"].shape == (32,)
        assert converted["image_mask"] == {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": False,
        }
        (directory / "ARMED").write_text("fake hardware test\n")
        action = np.asarray([0.4, -0.3, 0.2, 0.1, -0.2, 0.3, 1], dtype=np.float32)
        raw = np.broadcast_to(action, (1, 10, 7)).copy()
        sent = prepare_actions(raw, "real", "openpi", 10, 7)
        assert np.array_equal(raw, sent)
        for _ in range(duration_s):
            result = env.chunk_step(sent)
        assert result[3].any()
        with pytest.raises(RuntimeError, match="one episode"):
            env.reset()
    finally:
        env.close()
    evidence = [
        json.loads(line) for line in (directory / "wire.jsonl").read_text().splitlines()
    ]
    twists = [row for row in evidence if row["topic"] == ARM_TOPIC]
    nonzero = [row for row in twists if any(row["value"])]
    assert nonzero
    assert np.allclose(nonzero[0]["value"], action[:6])
    assert all(row["time"] - nonzero[0]["time"] < duration_s + 0.06 for row in nonzero)
    assert nonzero[-1]["time"] - nonzero[0]["time"] > duration_s - 0.3
    assert twists[-1]["value"] == [0] * 6
    assert len(twists) > 100
    assert {row["value"] for row in evidence if row["topic"] == GRIP_TOPIC} == {1}
    assert not env.env.envs[0].unwrapped.robot.is_connected


@pytest.mark.parametrize("suffix", ["", "_5s"])
def test_config_matches_sft_and_eval_only(monkeypatch, suffix):
    from evaluations.fr3_operator import validate_fr3_contract

    cfg = configuration(monkeypatch, "realworld_fr3_pi05_request_eval" + suffix)
    from rlinf.scheduler.cluster.config import ClusterConfig

    ClusterConfig.from_dict_cfg(cfg.cluster)
    assert "actor" not in cfg and "train" not in cfg.env and "algorithm" not in cfg
    sft = OmegaConf.load(
        ROOT
        / "examples/sft/config/realworld_sft_openpi_fr3_peg_request_full3000_wandb.yaml"
    )
    for key in (
        "config_name",
        "model_action_dim",
        "action_horizon",
        "discrete_state_input",
    ):
        assert cfg.rollout.model.openpi[key] == sft.actor.model.openpi[key]
    for key in ("model_path", "num_action_chunks", "action_dim", "openpi_data"):
        assert cfg.rollout.model[key] == sft.actor.model[key]
    if Path(cfg.runner.ckpt_path).is_file():
        validate_fr3_contract(cfg)
    cfg.rollout.model.rewrite_chunk_gripper = True
    with pytest.raises(ValueError, match="rewrite_chunk_gripper"):
        validate_fr3_contract(cfg)


def test_request_rejects_stale_issue_and_limits_gripper_edges():
    window = RequestWindow()
    with pytest.raises(ValueError, match="issue time"):
        window.submit([0] * 7, 1.0, 0.0)
    with pytest.raises(ValueError, match="issue time"):
        window.submit([0] * 7, 0.0, 0.1)
    assert window.started is None
    assert [window.submit([0] * 7, t, t) for t in (0, 0.1, 0.2)] == [None, None, 0]
    assert all(
        window.submit([0] * 6 + [1], t, t) is None
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    )
    assert window.submit([0] * 6 + [1], 1.01, 1.01) == 1


@pytest.mark.parametrize("failure", ["stale", "competitor", "workspace", "mode"])
@pytest.mark.parametrize("peer", [3, 5], indirect=True)
def test_active_peer_fails_to_neutral(peer, failure):
    (peer.session / "ARMED").write_text("fake hardware test\n")
    peer.command({"op": "reserve"})
    peer.spin()
    peer.command(
        {"op": "request", "value": [0.4] * 6 + [0.5], "issued": time.monotonic()}
    )
    peer.publish()
    assert peer.nodes[0].publishers[ARM_TOPIC][-1] == [0.4] * 6
    if failure == "stale":
        msg, stamp, source = peer.cache["pose"]
        peer.cache["pose"] = (msg, stamp - 0.3, source)
    elif failure == "competitor":
        peer.nodes[0].extra_publishers = 1
        peer.next_graph = 0
        peer.spin()
    elif failure == "workspace":
        peer.cache["pose"][0].pose.position.x += 0.081
    else:
        peer.cache["robot"][0].robot_mode = 4
    peer.publish()
    assert peer.nodes[0].publishers[ARM_TOPIC][-1] == [0] * 6
    assert peer.window.reason == "state_or_ownership_lost"


def test_subprocess_expires_and_stops_while_caller_is_idle(fake_session):
    from rlinf.robotics.parts.arms.fr3_ros2 import FR3ROS2RequestArm

    directory = Path(fake_session["session_dir"])
    arm = FR3ROS2RequestArm(
        **{
            key: value
            for key, value in fake_session.items()
            if key != "manual_episode_control_only"
        }
    )
    arm.connect()
    try:
        (directory / "ARMED").write_text("fake hardware test\n")
        arm.send_action({"request": [0.45] * 6 + [0.5]})
        time.sleep(0.16)
        rows = [
            json.loads(line)
            for line in (directory / "wire.jsonl").read_text().splitlines()
        ]
        assert any(any(row["value"]) for row in rows if row["topic"] == ARM_TOPIC)
        assert rows[-1]["value"] == [0] * 6
        arm.send_action({"request": [0.45] * 6 + [0.5]})
        arm.stop()
        time.sleep(0.04)
        rows = [
            json.loads(line)
            for line in (directory / "wire.jsonl").read_text().splitlines()
        ]
        assert rows[-1]["value"] == [0] * 6
        with pytest.raises(RuntimeError):
            arm.get_observation()
    finally:
        arm.disconnect()


def test_native_env_rejects_old_inference_observation(fake_session):
    from rlinf.envs.real.franka.request import FR3RequestEnv

    env = FR3RequestEnv(fake_session)
    directory = Path(fake_session["session_dir"])
    try:
        env.reset()
        env.on_action_chunk_begin()
        # The caller's clock is the authority for observation-to-action latency.
        time.sleep(1.01)
        (directory / "ARMED").write_text("fake hardware test\n")
        with pytest.raises(RuntimeError, match="inference observation"):
            env.step(np.zeros(7))
        assert (directory / "STOP").exists()
        assert not (directory / "wire.jsonl").exists()
    finally:
        env.close()


def test_operator_requires_tty_and_exact_confirmation(tmp_path, monkeypatch):
    import io

    from evaluations.fr3_operator import FR3OperatorSession

    monkeypatch.setattr("tempfile.mkdtemp", lambda **kwargs: str(tmp_path))
    session = FR3OperatorSession()
    monkeypatch.setattr(sys, "stdin", io.StringIO("RUN FR3 3S\n"))
    with pytest.raises(RuntimeError, match="interactive"):
        session.confirm()
    assert not (tmp_path / "ARMED").exists()
    session.stop()
    assert (tmp_path / "STOP").exists()


@pytest.mark.parametrize(
    "condition",
    [
        "observation",
        "driver_liveness",
        "parent_identity",
        "command_graph",
        "controller_ownership",
        "controller_response_freshness",
        "operator_stop",
        "operator_permission",
    ],
)
def test_health_diagnostics_identify_each_failed_condition(peer, condition):
    if condition == "observation":
        peer.cache.pop("pose")
    elif condition == "driver_liveness":
        peer.driver_pid = 2147483647
    elif condition == "parent_identity":
        peer.parent_pid = -1
    elif condition == "command_graph":
        peer.nodes[0].subscriber_count = 0
        peer.next_graph = 0
        peer.spin()
    elif condition == "controller_ownership":
        peer.controller_ok = False
    elif condition == "controller_response_freshness":
        peer.controller_at = time.monotonic() - 1.01
    elif condition == "operator_stop":
        (peer.session / "STOP").touch()
    else:
        peer.window.started = time.monotonic()
    assert not peer.healthy()
    assert set(peer.health_failures) == {condition}
    events = [json.loads(line) for line in Path(peer.log.name).read_text().splitlines()]
    status = [row for row in events if row["event"] == "health_check"][-1]
    assert status["healthy"] is False
    assert condition in status["failed_conditions"]
    if condition == "observation":
        error = status["failed_conditions"][condition]
        assert error["exception"] == "RuntimeError"
        assert '"pose": "missing"' in error["message"]
    elif condition == "command_graph":
        graph = status["failed_conditions"][condition]
        assert graph["topics"][ARM_TOPIC]["subscribers"] == 0
        assert graph["expected_subscribers"] == 1
    assert peer.pubs is None
    # Avoid creating a command window in fixture cleanup.
    peer.window.started = None


def test_missing_arm_diagnostics_preserve_initialization_refusal(peer):
    for name in ("pose", "joints", "wrench", "robot"):
        peer.cache.pop(name)
    peer.nodes[0].subscriber_count = 0
    peer.next_graph = 0
    peer.spin()
    # Keep the fake arm's state unavailable while the gripper remains fresh.
    for name in ("pose", "joints", "wrench", "robot"):
        peer.cache.pop(name, None)
    peer.controller_ok = False
    peer.controller_at = -float("inf")
    peer.controller_service_ready = False
    with pytest.raises(RuntimeError, match="unavailable:.*observation.*command_graph"):
        peer.command({"op": "observe"})
    assert peer.health_failures["controller_response_freshness"]["age_s"] is None
    assert peer.health_failures["controller_ownership"]["service_ready"] is False
    for name in ("pose", "joints", "wrench", "robot"):
        assert name in peer.health_failures["observation"]["message"]
    with pytest.raises(RuntimeError, match="Cannot reserve"):
        peer.command({"op": "reserve"})
    assert peer.pubs is None and peer.window.started is None


def test_health_diagnostics_throttle_and_record_recovery(peer):
    (peer.session / "STOP").touch()
    assert not peer.healthy()
    count = len(Path(peer.log.name).read_text().splitlines())
    assert not peer.healthy()
    assert len(Path(peer.log.name).read_text().splitlines()) == count
    peer._last_health_log -= 1.01
    assert not peer.healthy()
    assert len(Path(peer.log.name).read_text().splitlines()) == count + 1
    (peer.session / "STOP").unlink()
    assert peer.healthy()
    event = json.loads(Path(peer.log.name).read_text().splitlines()[-1])
    assert event["event"] == "health_check" and event["healthy"] is True
    assert event["failed_conditions"] == {}


def test_health_diagnostics_preserve_unexpected_observation_exception(peer):
    quaternion = peer.cache["pose"][0].pose.orientation
    previous = quaternion.x
    try:
        quaternion.x = float("nan")
        with pytest.raises(ValueError, match="finite"):
            peer.healthy()
        event = json.loads(Path(peer.log.name).read_text().splitlines()[-1])
        assert event["event"] == "health_exception"
        assert event["condition"] == "observation"
        assert event["exception"] == "ValueError"
    finally:
        quaternion.x = previous


def test_observation_timing_does_not_refresh_stale_action(fake_session):
    from rlinf.envs.real.franka.request import FR3RequestEnv

    env = FR3RequestEnv(fake_session)
    directory = Path(fake_session["session_dir"])
    try:
        env.reset()
        env.on_action_chunk_begin()
        time.sleep(1.01)
        with pytest.raises(RuntimeError, match="exceeded one second: age="):
            env.step(np.zeros(7))
        events = [
            json.loads(line)
            for line in (directory / "env_timing.jsonl").read_text().splitlines()
        ]
        acquired = next(row for row in events if row["event"] == "observation_acquired")
        admitted = next(row for row in events if row["event"] == "action_admission")
        assert admitted["received"] == acquired["received"]
        assert admitted["observation_age_s"] > 1.0
        assert not any(row["event"] == "request_issued" for row in events)
        assert not (directory / "wire.jsonl").exists()
    finally:
        env.close()
