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

"""CPU-only raw capture persistence checks; never open hardware."""

import json

import numpy as np
import pytest

from toolkits.realworld_check.fr3_raw_capture import RawCaptureWriter


def observation():
    return {
        "state": {
            "arm_joint_position": np.zeros(7, dtype=np.float32),
            "arm_joint_velocity": np.zeros(7, dtype=np.float32),
            "arm_joint_effort": np.zeros(7, dtype=np.float32),
            "tcp_pose": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            "tcp_force": np.zeros(3, dtype=np.float32),
            "tcp_torque": np.zeros(3, dtype=np.float32),
            "robotiq_knuckle_position": np.array([0.3], dtype=np.float32),
        },
        "frames": {
            "base": np.ones((2, 3, 3), dtype=np.uint8),
            "wrist": np.full((2, 4, 3), 42, dtype=np.uint8),
        },
        "metadata": {
            "schema": "fr3_robotiq_multimodal_raw_v1",
            "policy_ready": False,
            "source": {"joint_state": {"header": {"frame_id": "fr3_link0"}}},
        },
    }


def new_writer(tmp_path):
    return RawCaptureWriter(
        tmp_path / "pilot", task="stationary observation pilot",
        requested_samples=2, sample_rate_hz=10,
        camera_profiles={"base": {"serial_number": "A"}, "wrist": {"serial_number": "B"}},
        topic_names={"operator_twist": "/example_twist"}
    )


def test_raw_capture_roundtrip_retains_distinct_images_and_state(tmp_path):
    with new_writer(tmp_path) as writer:
        writer.write_sample(observation())
        writer.write_commands([
            {
                "name": "operator_twist",
                "message": {"linear": {"x": 0.1}, "angular": {"z": -0.1}},
                "received_monotonic_ns": 12345,
            },
            {
                "name": "operator_gripper_request",
                "message": {"data": 1.0},
                "received_monotonic_ns": 12346,
            }
        ])
        writer.finish(status="raw_recorded")

    root = tmp_path / "pilot"
    manifest = json.loads((root / "manifest.json").read_text())
    state = json.loads((root / "states.jsonl").read_text().strip())
    events = [json.loads(line) for line in (root / "operator_commands.jsonl").read_text().splitlines()]
    assert manifest["status"] == "raw_recorded"
    assert manifest["frames_saved"] == 1
    assert manifest["operator_command_events_saved"] == 2
    assert manifest["is_lerobot_dataset"] is False
    assert manifest["action_targets_7d_available"] is False
    assert state["training_action"] is None
    assert state["measured_state"]["robotiq_knuckle_position"] == pytest.approx([0.3])
    assert state["source_and_camera_metadata"]["source"]["joint_state"]["header"]["frame_id"] == "fr3_link0"
    np.testing.assert_array_equal(
        np.load(root / "frames/base/000000.npy"), np.ones((2, 3, 3), dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        np.load(root / "frames/wrist/000000.npy"), np.full((2, 4, 3), 42, dtype=np.uint8)
    )
    assert events[1]["received_monotonic_ns"] == 12346


def test_existing_directory_is_not_overwritten(tmp_path):
    with new_writer(tmp_path):
        pass
    with pytest.raises(FileExistsError):
        new_writer(tmp_path)


def test_dropped_commands_mark_capture_incomplete(tmp_path):
    with new_writer(tmp_path) as writer:
        writer.write_sample(observation())
        writer.write_commands([], dropped=3)
        writer.finish(status="raw_recorded")
    manifest = json.loads((tmp_path / "pilot/manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["operator_events_dropped"] == 3


def test_failure_marks_episode_interrupted(tmp_path):
    with pytest.raises(RuntimeError, match="stop"):
        with new_writer(tmp_path) as writer:
            writer.write_sample(observation())
            raise RuntimeError("stop")
    manifest = json.loads((tmp_path / "pilot/manifest.json").read_text())
    assert manifest["status"] == "interrupted"


def test_writer_rejects_training_ready_and_fake_command_names(tmp_path):
    with new_writer(tmp_path) as writer:
        obs = observation()
        obs["metadata"]["policy_ready"] = True
        with pytest.raises(ValueError, match="raw"):
            writer.write_sample(obs)
        with pytest.raises(ValueError, match="Unexpected"):
            writer.write_commands([{"name": "fake_policy_action", "message": {}}])
