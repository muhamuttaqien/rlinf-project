"""Offline tests for diagnostic FR3 observed-motion proxy (never SFT labels)."""

import json

import numpy as np
import pytest

from toolkits.realworld_check.fr3_observed_action_proxy import (
    analyze_episode, base_frame_rotvec,
)
from toolkits.realworld_check.fr3_raw_capture import RAW_SCHEMA


def _make_episode(tmp_path, *, stamps=(1_000_000_000, 1_100_000_000, 1_200_000_000),
                  poses=None):
    root = tmp_path / "episode_0001"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "schema": RAW_SCHEMA, "status": "raw_recorded",
        "frames_saved": len(stamps),
    }))
    if poses is None:
        poses = [[0.01 * i, 0, 0, 0, 0, 0, 1] for i in range(len(stamps))]
    with (root / "states.jsonl").open("w") as stream:
        for i, (stamp, pose) in enumerate(zip(stamps, poses, strict=True)):
            stream.write(json.dumps({
                "frame_index": i, "recorded_monotonic_ns": stamp,
                "training_action": None,
                "measured_state": {
                    "tcp_pose": pose, "robotiq_knuckle_position": [0.1 + i * 0.1],
                },
                "source_and_camera_metadata": {
                    "source": {"tcp_pose": {"received_monotonic_ns": stamp}}
                }
            }) + "\n")
    return root


def test_proxy_tracks_next_pose_and_measured_gripper(tmp_path):
    root = _make_episode(tmp_path)
    report, rows = analyze_episode(root)
    assert report["observed_transition_proxies"] == 2
    assert report["is_controller_accepted_action"] is False
    assert report["sft_ready"] is False
    np.testing.assert_allclose(
        rows[0]["observed_transition_proxy_7d"],
        [0.01, 0, 0, 0, 0, 0, 0.2], atol=1e-8,
    )
    assert report["rejected_time_intervals"] == 0


def test_quaternion_sign_flip_not_false_rotation():
    q = np.array([0., 0., 0., 1.])
    np.testing.assert_allclose(base_frame_rotvec(q, -q), [0, 0, 0], atol=1e-10)


def test_rotation_about_base_z():
    h = np.pi / 4
    q0 = [0, 0, 0, 1]
    q1 = [0, 0, np.sin(h), np.cos(h)]
    np.testing.assert_allclose(base_frame_rotvec(q0, q1), [0, 0, np.pi / 2], atol=1e-8)


def test_bad_timing_rejected_without_fake_action(tmp_path):
    root = _make_episode(tmp_path, stamps=(1_000_000_000, 1_100_000_000,
                                          2_100_000_000))
    report, rows = analyze_episode(root)
    assert len(rows) == 1
    assert report["rejected_time_intervals"] == 1


def test_nonmonotonic_all_timing_blocks_output(tmp_path):
    root = _make_episode(tmp_path, stamps=(1_000_000_000, 1_000_000_000,
                                          999_000_000))
    with pytest.raises(ValueError, match="No valid adjacent"):
        analyze_episode(root)
