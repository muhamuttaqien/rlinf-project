"""Offline-only FR3 observation preparation: no actions or hardware."""

import json

import numpy as np
import pytest

from toolkits.realworld_check import fr3_pi05_training_prep as prep


def _raw_fixture(tmp_path, *, inject_action=False):
    raw = tmp_path / "raw"
    raw.mkdir()
    measured = {
        "tcp_pose": [0.4, 0.1, 0.5, 0, 0, 0, 1],
        "arm_joint_position": [0] * 7,
        "tcp_force": [0, 0, 0],
        "robotiq_knuckle_position": [0.2],
        "tcp_torque": [0, 0, 0],
    }
    rows = [
        dict(frame_index=i, recorded_monotonic_ns=1000+i,
             training_action=[0]*7 if inject_action else None,
             operator_command_labels=None, measured_state=measured)
        for i in range(2)
    ]
    (raw / "states.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in rows)
    )
    return raw


def test_two_real_state_shapes_but_never_training_ready(monkeypatch, tmp_path):
    raw = _raw_fixture(tmp_path)
    monkeypatch.setattr(prep, "audit_raw_episode", lambda _: (
        {"sample_count": 2, "camera_serials": {"base": "a", "wrist": "b"}}, {}
    ))
    report, states, stamps = prep.prepare_observations(raw)
    assert states.shape == (2, 19)
    assert states.dtype == np.float32
    assert stamps.tolist() == [1000, 1001]
    assert report["training_actions_present"] is False
    assert report["real_sft_ready"] is False
    assert report["recap_ready"] is False
    assert report["robot_commands_sent"] is False


def test_no_fake_action_labels(monkeypatch, tmp_path):
    raw = _raw_fixture(tmp_path, inject_action=True)
    monkeypatch.setattr(prep, "audit_raw_episode", lambda _: (
        {"sample_count": 2, "camera_serials": {"base": "a", "wrist": "b"}}, {}
    ))
    with pytest.raises(ValueError, match="training action"):
        prep.prepare_observations(raw)


def test_existing_output_not_overwritten(monkeypatch, tmp_path):
    raw = _raw_fixture(tmp_path)
    out = tmp_path / "already_exists"
    out.mkdir()
    monkeypatch.setattr(prep, "audit_raw_episode", lambda _: (
        {"sample_count": 2, "camera_serials": {"base": "a", "wrist": "b"}}, {}
    ))
    monkeypatch.setattr(
        "sys.argv", ["prep", "--raw", str(raw), "--output", str(out)]
    )
    with pytest.raises(FileExistsError):
        prep.main()
