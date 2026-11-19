"""Tests for offline FR3 browser dashboard. No robot connection."""
import json
from pathlib import Path

import numpy as np
import pytest

from toolkits.realworld_check.fr3_episode_web_dashboard import EpisodeStore


def make_episode(root: Path):
    ep = root / "episode_0001"
    (ep / "frames" / "base").mkdir(parents=True)
    (ep / "frames" / "wrist").mkdir(parents=True)
    (ep / "manifest.json").write_text(json.dumps({
        "status": "raw_recorded", "task": "peg insertion",
        "frames_saved": 2, "sample_rate_hz_requested": 10,
    }))
    (ep / "episode_review.json").write_text(json.dumps({"operator_outcome": "success"}))
    rows = []
    for i in range(2):
        paths = {}
        for camera in ("base", "wrist"):
            rel = f"frames/{camera}/{i:06d}.npy"
            array = np.zeros((2, 3, 3), dtype=np.uint8)
            array[0, 0] = [10, 20, 30]  # BGR
            np.save(ep / rel, array)
            paths[camera] = rel
        rows.append({
            "frame_index": i, "recorded_monotonic_ns": 1000000000 + i * 100000000,
            "image_files_bgr8": paths,
            "measured_state": {
                "tcp_pose": [0.1, 0.2 + i, 0.3, 0, 0, 0, 1],
                "arm_joint_position": [0]*7,
                "arm_joint_velocity": [0]*7,
                "arm_joint_effort": [0]*7,
                "tcp_force": [0, 1, 2],
                "tcp_torque": [0, 0, 0],
                "robotiq_knuckle_position": [0.1],
            }
        })
    (ep / "states.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))
    (ep / "operator_commands.jsonl").write_text(json.dumps({
        "name": "operator_twist", "received_monotonic_ns": 1050000000,
        "message": {"linear": {"x": 0.5, "y": 0, "z": 0},
                    "angular": {"x": 0, "y": 0, "z": 0}}
    })+"\n")
    return ep


def test_store_reads_episode_and_frames(tmp_path):
    make_episode(tmp_path)
    store = EpisodeStore(tmp_path)
    summary = store.episodes()
    assert summary["episode_count"] == 1
    assert summary["operator_successes"] == 1
    data, _ = store.load("episode_0001")
    assert data["frames"] == 2
    assert data["times_s"] == [0.0, 0.1]
    assert data["series"]["tcp_y"] == pytest.approx([0.2, 1.2])
    assert data["series"]["op_lin_x"] == [None, 0.5]
    assert data["operator_events"]["twist"]["times_s"] == [0.05]
    assert data["operator_events"]["twist"]["values"] == [0.5]
    assert data["operator_events"]["gripper"]["times_s"] == []
    assert store.frame_jpeg("episode_0001", "base", 0).startswith(b"\xff\xd8")


def test_rejects_invalid_paths_and_indexes(tmp_path):
    make_episode(tmp_path)
    store = EpisodeStore(tmp_path)
    with pytest.raises(ValueError):
        store.episode_dir("../episode_0001")
    with pytest.raises(IndexError):
        store.frame_jpeg("episode_0001", "base", 12)
    with pytest.raises(ValueError):
        store.frame_jpeg("episode_0001", "other", 0)


def test_rejects_symlink_episode(tmp_path):
    outside = tmp_path.parent / "fake_episode"
    outside.mkdir(exist_ok=True)
    (tmp_path / "episode_0099").symlink_to(outside, target_is_directory=True)
    store = EpisodeStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.episode_dir("episode_0099")
    assert store.episodes()["episode_count"] == 0
