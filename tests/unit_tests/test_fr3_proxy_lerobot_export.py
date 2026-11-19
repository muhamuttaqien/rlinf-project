"""Offline validation tests for experimental observed-proxy LeRobot export."""

import json
from pathlib import Path

import numpy as np
import pytest

from toolkits.realworld_check.fr3_observed_action_proxy import (
    SCHEMA, analyze_episode,
)
from toolkits.realworld_check.fr3_proxy_lerobot_export import validate_episode
from toolkits.realworld_check.fr3_raw_capture import RAW_SCHEMA


def _fixture(tmp_path):
    raw = tmp_path / "episode_0001"
    raw.mkdir()
    (raw / "manifest.json").write_text(json.dumps({
        "schema": RAW_SCHEMA, "status": "raw_recorded",
        "frames_saved": 2, "task": "FR3 peg insertion",
        "camera_profiles": {
            "base": {"resolution": [2, 2]},
            "wrist": {"resolution": [2, 2]},
        },
        "sample_rate_hz_requested": 10,
    }))
    (raw / "episode_review.json").write_text(json.dumps({
        "operator_outcome": "success", "capture_status": "raw_recorded",
    }))
    with (raw / "states.jsonl").open("w") as f:
        for i in range(2):
            f.write(json.dumps({
                "frame_index": i,
                "recorded_monotonic_ns": 1_000_000_000 + 100_000_000 * i,
                "training_action": None,
                "measured_state": {
                    "tcp_pose": [i * .01, 0, 0, 0, 0, 0, 1],
                    "robotiq_knuckle_position": [0.1],
                },
                "source_and_camera_metadata": {
                    "source": {"tcp_pose": {
                        "received_monotonic_ns": 1_000_000_000 + 100_000_000 * i,
                    }}
                },
            }) + "\n")
    summary, records = analyze_episode(raw)
    proxy = tmp_path / "episode_0001.json"
    proxy.write_text(json.dumps({
        "schema": SCHEMA, "summary": summary, "records": records,
    }))
    return raw, proxy


def test_valid_observed_proxy_is_diagnostic_only(tmp_path):
    raw, proxy = _fixture(tmp_path)
    result = validate_episode(raw, proxy)
    assert result["transitions"] == 1


def test_wrong_proxy_action_refused(tmp_path):
    raw, proxy = _fixture(tmp_path)
    data = json.loads(proxy.read_text())
    data["records"][0]["observed_transition_proxy_7d"][0] = 0.5
    proxy.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        validate_episode(raw, proxy)


def test_failed_episode_refused(tmp_path):
    raw, proxy = _fixture(tmp_path)
    (raw / "episode_review.json").write_text(json.dumps({
        "operator_outcome": "failure", "capture_status": "raw_recorded",
    }))
    with pytest.raises(ValueError, match="not a reviewed"):
        validate_episode(raw, proxy)


def test_stale_raw_manifest_refused(tmp_path):
    raw, proxy = _fixture(tmp_path)
    manifest = json.loads((raw / "manifest.json").read_text())
    manifest["task"] = "changed"
    (raw / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="stale/mismatched"):
        validate_episode(raw, proxy)
