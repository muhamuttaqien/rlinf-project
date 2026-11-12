"""Offline tests for SpaceMouse request alignment, never robot actions."""
import json

import pytest

from toolkits.realworld_check.fr3_command_request_alignment import (
    _mean_held_twist, _grip, _twist, align_episode, build_dataset,
)
from toolkits.realworld_check.fr3_raw_capture import (
    RAW_SCHEMA, TWIST_TOPIC, GRIPPER_REQUEST_TOPIC,
)


def _twist_event(stamp, x):
    return {
        "name": "operator_twist", "topic": TWIST_TOPIC,
        "message_type": "geometry_msgs/msg/Twist",
        "received_monotonic_ns": stamp,
        "message": {
            "linear": {"x": x, "y": 0, "z": 0},
            "angular": {"x": 0, "y": 0, "z": 0},
        },
    }


def _grip_event(stamp, value):
    return {
        "name": "operator_gripper_request", "topic": GRIPPER_REQUEST_TOPIC,
        "message_type": "std_msgs/msg/Float32",
        "received_monotonic_ns": stamp, "message": {"data": value},
    }


def _episode(tmp_path, include_grip=True):
    root = tmp_path / "raw"
    ep = root / "episode_0001"
    for role in ("base", "wrist"):
        (ep / "frames" / role).mkdir(parents=True)
    stamps = [1_000_000_000, 1_100_000_000, 1_200_000_000]
    rows = []
    for i, stamp in enumerate(stamps):
        images = {
            role: f"frames/{role}/{i:06d}.npy"
            for role in ("base", "wrist")
        }
        for relative in images.values():
            (ep / relative).write_bytes(b"reference-only")
        rows.append({
            "frame_index": i,
            "recorded_monotonic_ns": stamp + 20_000_000,
            "training_action": None,
            "measured_state": {
                "tcp_pose": [0, 0, 0, 0, 0, 0, 1],
                "arm_joint_position": [0] * 7,
                "tcp_force": [0] * 3,
                "tcp_torque": [0] * 3,
                "robotiq_knuckle_position": [0.1],
            },
            "image_files_bgr8": images,
            "source_and_camera_metadata": {
                "cameras": {
                    "base": {"received_monotonic_ns": stamp - 1_000_000},
                    "wrist": {"received_monotonic_ns": stamp},
                },
            },
        })
    events = [
        _twist_event(990_000_000, 0.5),
        _twist_event(1_050_000_000, 1.0),
        _twist_event(1_150_000_000, -0.5),
    ]
    if include_grip:
        events.insert(0, _grip_event(995_000_000, 1.0))
    (ep / "manifest.json").write_text(json.dumps({
        "schema": RAW_SCHEMA, "status": "raw_recorded",
        "frames_saved": len(rows),
        "operator_command_events_saved": len(events),
        "operator_events_dropped": 0,
        "executed_arm_commands_available": False,
    }))
    (ep / "states.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in rows)
    )
    (ep / "operator_commands.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in events)
    )
    return root, ep


def test_mean_zoh_request_does_not_sum_messages():
    v, coverage = _mean_held_twist(
        100, 200,
        [90, 150],
        [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        200,
    )
    assert coverage == 1
    assert v[0] == pytest.approx(0.5)


def test_stale_request_not_fabricated():
    v, coverage = _mean_held_twist(
        100, 200, [90], [[1, 0, 0, 0, 0, 0]], 20,
    )
    assert coverage == pytest.approx(0.1)
    assert v[0] == pytest.approx(1)


def test_reject_unnormalized_and_invalid_gripper():
    with pytest.raises(ValueError, match="normalized"):
        _twist(_twist_event(100, 2.0))
    with pytest.raises(ValueError, match="binary"):
        _grip(_grip_event(100, 0.5))


def test_alignment_keeps_original_images_and_requests(tmp_path):
    raw_root, ep = _episode(tmp_path)
    report, rows = align_episode(ep)
    assert report["counts"]["candidate_7d_available"] == 2
    assert len(rows) == 2
    assert rows[0]["candidate_request_action_7d"][0] == pytest.approx(0.75)
    assert rows[0]["candidate_request_action_7d"][-1] == 1.0
    assert rows[1]["candidate_request_action_7d"][0] == pytest.approx(0.25)
    assert rows[0]["images_bgr8_npy"]["base"] == "frames/base/000000.npy"

    dest = tmp_path / "candidate"
    summary = build_dataset(raw_root, dest, expected_episodes=1)
    assert summary["transitions"] == 2
    assert summary["sft_ready"] is False
    assert summary["recap_ready"] is False
    assert summary["physical_execution_approved"] is False
    assert (dest / "episodes" / "episode_0001.jsonl").is_file()
    with pytest.raises(FileExistsError):
        build_dataset(raw_root, dest, expected_episodes=1)


def test_unknown_gripper_is_null_not_measured_knuckle(tmp_path):
    _, ep = _episode(tmp_path, include_grip=False)
    report, rows = align_episode(ep)
    assert report["counts"].get("candidate_7d_available", 0) == 0
    assert all(row["candidate_request_action_7d"] is None for row in rows)
    assert all(row["gripper_request_open_fraction"] is None for row in rows)
