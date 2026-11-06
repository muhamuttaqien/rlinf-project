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

"""CPU-only tests for raw FR3, Robotiq and dual-camera observations."""

import numpy as np
import pytest

from toolkits.realworld_check.fr3_multimodal import assemble_fr3_multimodal


def record(name, message):
    return {"status": "available", "source": {
        "event": "state", "name": name, "message": message,
        "received_monotonic_ns": 1_000_000_000,
        "received_unix_ns": 1_792_000_000_000_000_000,
        "topic": "/" + name,
    }}


def sample():
    names = [f"fr3_joint{i}" for i in range(1, 8)]
    arm = {"states": {
        "joint_state": record("joint_state", {
            "header": {"stamp": {"sec": 42, "nanosec": 5}, "frame_id": "fr3_link0"},
            "name": names[::-1],
            "position": list(range(7)), "velocity": [0.1] * 7, "effort": [1.0] * 7,
        }),
        "tcp_pose": record("tcp_pose", {
            "pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3},
                     "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
        }),
        "external_wrench": record("external_wrench", {
            "wrench": {"force": {"x": 1, "y": 2, "z": 3},
                       "torque": {"x": 4, "y": 5, "z": 6}},
        }),
    }}
    gripper = {"states": {"gripper_state": record("gripper_state", {
        "name": ["robotiq_85_left_knuckle_joint"], "position": [0.7754],
    })}}
    base = np.zeros((2, 3, 3), dtype=np.uint8)
    base[0, 0] = [10, 20, 30]
    cameras = {
        "base": {"frame": base, "serial_number": "D435i",
                 "encoding": "bgr8", "received_monotonic_ns": 1_000_000_000},
        "wrist": {"frame": np.ones((2, 3, 3), dtype=np.uint8),
                  "serial_number": "D405", "encoding": "bgr8",
                  "received_monotonic_ns": 1_025_000_000},
    }
    return arm, gripper, cameras


def test_valid_sample_preserves_source_and_converts_bgr_explicitly():
    arm, gripper, cameras = sample()
    result = assemble_fr3_multimodal(arm, gripper, cameras)
    assert result["metadata"]["policy_ready"] is False
    assert result["state"]["arm_joint_position"].tolist() == list(range(6, -1, -1))
    assert result["state"]["tcp_pose"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0, 0, 0, 1])
    assert result["state"]["tcp_force"].tolist() == pytest.approx([1, 2, 3])
    assert result["state"]["robotiq_knuckle_position"][0] == pytest.approx(0.7754)
    assert result["frames"]["base"][0, 0].tolist() == [10, 20, 30]
    assert result["policy_images_rgb"]["base"][0, 0].tolist() == [30, 20, 10]
    assert result["metadata"]["source"]["joint_state"]["header"]["stamp"]["sec"] == 42
    assert result["metadata"]["camera_receipt_skew_s"] == pytest.approx(0.025)
    cameras["base"]["frame"][0, 0] = [255, 255, 255]
    assert result["frames"]["base"][0, 0].tolist() == [10, 20, 30]
    assert "observation/state" not in result


@pytest.mark.parametrize("name", ["joint_state", "tcp_pose", "external_wrench"])
def test_missing_arm_source_is_rejected(name):
    arm, gripper, cameras = sample()
    arm["states"][name]["status"] = "unavailable"
    with pytest.raises(ValueError, match="unavailable"):
        assemble_fr3_multimodal(arm, gripper, cameras)


def test_missing_gripper_is_rejected():
    arm, gripper, cameras = sample()
    gripper["states"]["gripper_state"]["status"] = "unavailable"
    with pytest.raises(ValueError, match="gripper_state unavailable"):
        assemble_fr3_multimodal(arm, gripper, cameras)


def test_missing_arm_joint_is_rejected():
    arm, gripper, cameras = sample()
    arm["states"]["joint_state"]["source"]["message"]["name"][0] = "wrong_joint"
    with pytest.raises(ValueError, match="Missing FR3 joints"):
        assemble_fr3_multimodal(arm, gripper, cameras)


def test_nonfinite_input_is_rejected():
    arm, gripper, cameras = sample()
    arm["states"]["joint_state"]["source"]["message"]["velocity"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        assemble_fr3_multimodal(arm, gripper, cameras)


def test_missing_wrist_is_rejected():
    arm, gripper, cameras = sample()
    cameras.pop("wrist")
    with pytest.raises(ValueError, match="Exactly base and wrist"):
        assemble_fr3_multimodal(arm, gripper, cameras)


def test_camera_skew_and_duplicate_serial_are_rejected():
    arm, gripper, cameras = sample()
    cameras["wrist"]["received_monotonic_ns"] += 200_000_000
    with pytest.raises(ValueError, match="skew"):
        assemble_fr3_multimodal(arm, gripper, cameras)
    cameras["wrist"]["received_monotonic_ns"] = 1_025_000_000
    cameras["wrist"]["serial_number"] = cameras["base"]["serial_number"]
    with pytest.raises(ValueError, match="distinct"):
        assemble_fr3_multimodal(arm, gripper, cameras)


def test_camera_encoding_is_explicit():
    arm, gripper, cameras = sample()
    cameras["base"]["encoding"] = "rgb8"
    with pytest.raises(ValueError, match="bgr8"):
        assemble_fr3_multimodal(arm, gripper, cameras)
