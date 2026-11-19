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

"""Causal FR3 frequency conversion and genuine raw-data roundtrip contracts."""

import json

import numpy as np
import pytest

from toolkits.realworld_check.fr3_frequency_dataset import (
    PERIOD_NS,
    export_split,
    fixed_grid,
    held_integral,
    process_episode,
    split_assignments,
    validate_split,
)
from toolkits.realworld_check.fr3_raw_capture import GRIPPER_REQUEST_TOPIC, TWIST_TOPIC


def fixture_episode(root):
    ep = root / "episode_0001"
    for role in ("base", "wrist"):
        (ep / "frames" / role).mkdir(parents=True)
    start = 10_000_000_000
    rows = []
    for i in range(22):
        received = start + i * PERIOD_NS
        metadata = {
            "arm_joint_names": [f"fr3_joint{j}" for j in range(1, 8)],
            "gripper_joint_name": "robotiq_85_left_knuckle_joint",
            "gripper_units": "knuckle_radians_not_metric_width",
            "source": {
                k: {
                    "received_monotonic_ns": received,
                    "received_unix_ns": received,
                    "header": {
                        "stamp": {
                            "sec": received // 10**9,
                            "nanosec": received % 10**9,
                        },
                        "frame_id": "base",
                    },
                }
                for k in ("joint_state", "tcp_pose", "external_wrench", "gripper_state")
            },
            "cameras": {
                role: {"received_monotonic_ns": received - 10_000_000}
                for role in ("base", "wrist")
            },
        }
        files = {role: f"frames/{role}/{i:06d}.npy" for role in ("base", "wrist")}
        for role, filename in files.items():
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            image[:] = [i, 30, 70]
            np.save(ep / filename, image, allow_pickle=False)
        rows.append(
            {
                "frame_index": i,
                "training_action": None,
                "recorded_monotonic_ns": received + 20_000_000 + i * 100_000,
                "source_and_camera_metadata": metadata,
                "image_files_bgr8": files,
                "measured_state": {
                    "tcp_pose": [i / 100, 0, 0, 0, 0, 0, 1],
                    "arm_joint_position": [0] * 7,
                    "tcp_force": [1, 2, 3],
                    "tcp_torque": [0, 0, 1],
                    "robotiq_knuckle_position": [0.785],
                },
            }
        )
    events = []
    for i in range(231):
        value = 0 if i < 55 else 0.5
        events.append(
            {
                "name": "operator_twist",
                "topic": TWIST_TOPIC,
                "message_type": "geometry_msgs/msg/Twist",
                "units": {
                    "linear": "normalized_controller_input_not_m_per_s",
                    "angular": "normalized_controller_input_not_rad_per_s",
                },
                "message": {
                    "linear": {"x": value, "y": 0, "z": 0},
                    "angular": {"x": 0, "y": 0, "z": 0},
                },
                "received_monotonic_ns": start + i * 10_000_000,
            }
        )
    for relative, value in ((250_000_000, 0.0), (1_050_000_000, 1.0)):
        events.append(
            {
                "name": "operator_gripper_request",
                "topic": GRIPPER_REQUEST_TOPIC,
                "message_type": "std_msgs/msg/Float32",
                "message": {"data": value},
                "received_monotonic_ns": start + relative,
            }
        )
    events.sort(key=lambda x: x["received_monotonic_ns"])
    manifest = {
        "schema": "fr3_robotiq_raw_capture_v1",
        "status": "raw_recorded",
        "operator_events_dropped": 0,
        "failures": {},
        "frames_saved": len(rows),
        "operator_command_events_saved": len(events),
        "task": "FR3 peg insertion",
        "camera_profiles": {
            role: {"resolution": [2, 2], "fps": 30} for role in ("base", "wrist")
        },
    }
    review = {
        "operator_outcome": "success",
        "operator_reviewed": True,
        "capture_end_workstation_monotonic_ns": start + 2_250_000_000,
    }
    for name, value in (("manifest.json", manifest), ("episode_review.json", review)):
        (ep / name).write_text(json.dumps(value))
    for name, value in (("states.jsonl", rows), ("operator_commands.jsonl", events)):
        (ep / name).write_text("".join(json.dumps(r) + "\n" for r in value))
    return ep, rows, events


def test_fixed_grid_never_uses_future_or_repeated_observations():
    stamps = [1_000_000_000 + i * PERIOD_NS + i * 100_000 for i in range(22)]
    grid = fixed_grid(stamps, stamps[-1] + PERIOD_NS)
    assert np.all(np.diff(grid) == PERIOD_NS)
    assert np.all(np.asarray(stamps[: len(grid)]) <= grid)
    assert np.array_equal(
        np.searchsorted(stamps, grid, side="right") - 1, np.arange(len(grid))
    )


@pytest.mark.parametrize("stamps", [[1, 1], [2, 1], [1, 200_000_001, 400_000_001]])
def test_invalid_clock_or_large_gap_refused(stamps):
    with pytest.raises(ValueError):
        fixed_grid(stamps, 2_000_000_000)


def test_held_requests_are_averaged_not_summed():
    result = held_integral(
        1_000_000_000, 1_100_000_000, [1_000_000_000, 1_050_000_000], [[0] * 6, [1] * 6]
    )
    assert np.allclose(result, [0.05] * 6)
    assert np.allclose(result / 0.1, [0.5] * 6)


def test_causal_labels_preserve_motion_and_discrete_events(tmp_path):
    ep, source, _ = fixture_episode(tmp_path)
    report, rows, _ = process_episode(ep, "train")
    assert report["request_integral_error_float64"] < 1e-12
    assert all(r["source_recorded_ns"] <= r["grid_start_ns"] for r in rows)
    assert report["unknown_gripper_prefix_bins"] > 0
    assert [e["request"] for e in report["gripper_events"]] == [0, 1]
    assert all(0 <= e["delay_s"] < 0.1 for e in report["gripper_events"])
    assert rows[0]["actions"][6] == 0 and rows[0]["state"][17] == pytest.approx(0.785)
    for row in rows:
        assert row["state"][:7] == pytest.approx(
            source[row["source_frame_index"]]["measured_state"]["tcp_pose"]
        )
    # The command step occurs inside one 100 ms bin: its target is time weighted.
    mixed = [r["actions"][0] for r in rows if 0 < r["actions"][0] < 0.5]
    assert len(mixed) == 1


def test_export_roundtrip_including_rgb_and_source_clock(tmp_path):
    ep, _, _ = fixture_episode(tmp_path / "raw")
    report, rows, _ = process_episode(ep, "train")
    before = {p: p.read_bytes() for p in ep.rglob("*") if p.is_file()}
    output = tmp_path / "converted"
    export_split(ep.parent, output, "train", [report], {ep.name: rows}, threads=2)
    checked = validate_split(ep.parent, output, {ep.name: rows})
    assert checked["all_rgb_pixels_equal"] and checked["frames"] == len(rows)
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize(
    "problem", ["metric_units", "stale_request_gap", "future_receipt", "merged_buttons"]
)
def test_unreliable_conversion_refused(tmp_path, problem):
    ep, rows, events = fixture_episode(tmp_path)
    if problem == "metric_units":
        events[0]["units"] = {"linear": "m/s", "angular": "rad/s"}
    elif problem == "stale_request_gap":
        events = [
            e
            for e in events
            if not (10_600_000_000 < e["received_monotonic_ns"] < 10_850_000_000)
        ]
        manifest = json.loads((ep / "manifest.json").read_text())
        manifest["operator_command_events_saved"] = len(events)
        (ep / "manifest.json").write_text(json.dumps(manifest))
    elif problem == "future_receipt":
        rows[0]["source_and_camera_metadata"]["cameras"]["base"][
            "received_monotonic_ns"
        ] += 200_000_000
    else:
        buttons = [e for e in events if e["name"] == "operator_gripper_request"]
        buttons[1]["received_monotonic_ns"] = buttons[0]["received_monotonic_ns"] + 1
        events.sort(key=lambda x: x["received_monotonic_ns"])
    (ep / "states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (ep / "operator_commands.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in events)
    )
    with pytest.raises(ValueError):
        process_episode(ep, "train")


def test_split_overlap_is_rejected(tmp_path):
    for split in ("train", "validation"):
        p = tmp_path / split
        p.mkdir()
        (p / "fr3_command_request_provenance.json").write_text(
            json.dumps(
                {
                    "source_split": split,
                    "source_episodes": 1,
                    "segments": [{"episode": "episode_0001"}],
                }
            )
        )
    with pytest.raises(ValueError, match="leakage"):
        split_assignments(tmp_path / "train", tmp_path / "validation")
