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

"""Offline 4C raw-to-RGB preflight tests; no ROS or hardware access."""

import json

import numpy as np
import pytest

from toolkits.realworld_check.fr3_lerobot_preflight import audit_raw_episode


def episode(tmp_path):
    root = tmp_path / "raw"
    (root / "frames/base").mkdir(parents=True)
    (root / "frames/wrist").mkdir(parents=True)
    base = np.array([[[10, 20, 30], [11, 21, 31]]], dtype=np.uint8)
    wrist = np.array([[[40, 50, 60]]], dtype=np.uint8)
    np.save(root / "frames/base/000000.npy", base)
    np.save(root / "frames/wrist/000000.npy", wrist)
    manifest = {
        "schema": "fr3_robotiq_raw_capture_v1",
        "status": "raw_recorded",
        "task": "stationary test",
        "is_lerobot_dataset": False,
        "policy_ready": False,
        "action_targets_7d_available": False,
        "executed_arm_commands_available": False,
        "operator_events_dropped": 0,
        "failures": {},
        "frames_saved": 1,
        "operator_command_events_saved": 1,
        "camera_profiles": {
            "base": {"resolution": [2, 1], "serial_number": "A"},
            "wrist": {"resolution": [1, 1], "serial_number": "B"},
        },
    }
    state = {
        "frame_index": 0,
        "recorded_monotonic_ns": 123,
        "measured_state": {
            "arm_joint_position": [0] * 7,
            "arm_joint_velocity": [0] * 7,
            "arm_joint_effort": [0] * 7,
            "tcp_pose": [0, 0, 0, 0, 0, 0, 1],
            "tcp_force": [0] * 3,
            "tcp_torque": [0] * 3,
            "robotiq_knuckle_position": [0.3],
        },
        "image_files_bgr8": {
            "base": "frames/base/000000.npy",
            "wrist": "frames/wrist/000000.npy",
        },
        "training_action": None,
        "source_and_camera_metadata": {
            "policy_ready": False,
            "cameras": {
                "base": {
                    "raw_encoding": "bgr8",
                    "serial_number": "A",
                    "received_monotonic_ns": 120,
                },
                "wrist": {
                    "raw_encoding": "bgr8",
                    "serial_number": "B",
                    "received_monotonic_ns": 121,
                },
            },
        },
    }
    event = {
        "name": "operator_twist",
        "received_monotonic_ns": 125,
        "message": {"linear": {"x": 0.0}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "states.jsonl").write_text(json.dumps(state) + "\n")
    (root / "operator_commands.jsonl").write_text(json.dumps(event) + "\n")
    return root


def test_preflight_accepts_two_actual_rgb_views_but_not_training_readiness(tmp_path):
    root = episode(tmp_path)
    report, previews = audit_raw_episode(root)
    assert report["raw_capture_valid"]
    assert report["sample_count"] == 1
    assert report["sft_export_ready"] is False
    assert report["recap_export_ready"] is False
    assert len(report["blocking_requirements"]) == 5
    np.testing.assert_array_equal(previews["image"][0, 0], [30, 20, 10])
    np.testing.assert_array_equal(previews["extra_view_image"][0, 0], [60, 50, 40])


def test_missing_wrist_frame_rejected(tmp_path):
    root = episode(tmp_path)
    (root / "frames/wrist/000000.npy").unlink()
    with pytest.raises(ValueError, match="wrist"):
        audit_raw_episode(root)


def test_missing_camera_provenance_rejected(tmp_path):
    root = episode(tmp_path)
    path = root / "states.jsonl"
    state = json.loads(path.read_text())
    del state["source_and_camera_metadata"]["cameras"]["wrist"]
    path.write_text(json.dumps(state) + "\n")
    with pytest.raises(ValueError, match="provenance"):
        audit_raw_episode(root)


def test_nonfinite_state_rejected(tmp_path):
    root = episode(tmp_path)
    path = root / "states.jsonl"
    state = json.loads(path.read_text())
    state["measured_state"]["tcp_force"][0] = float("nan")
    path.write_text(json.dumps(state) + "\n")
    with pytest.raises(ValueError, match="finite"):
        audit_raw_episode(root)


def test_dropped_events_rejected(tmp_path):
    root = episode(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["operator_events_dropped"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dropped"):
        audit_raw_episode(root)


def test_training_label_or_claim_rejected(tmp_path):
    root = episode(tmp_path)
    path = root / "states.jsonl"
    state = json.loads(path.read_text())
    state["training_action"] = [0] * 7
    path.write_text(json.dumps(state) + "\n")
    with pytest.raises(ValueError, match="training actions"):
        audit_raw_episode(root)


def test_camera_serial_mismatch_rejected(tmp_path):
    root = episode(tmp_path)
    path = root / "states.jsonl"
    state = json.loads(path.read_text())
    state["source_and_camera_metadata"]["cameras"]["base"]["serial_number"] = "other"
    path.write_text(json.dumps(state) + "\n")
    with pytest.raises(ValueError, match="encoding/serial"):
        audit_raw_episode(root)
