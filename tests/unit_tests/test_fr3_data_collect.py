# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline-only FR3 manual episode persistence tests."""

import json

import numpy as np
import pytest

from toolkits.realworld_check.fr3_data_collect import (
    REVIEW_SCHEMA,
    capture_episode,
    discard_aborted_episode,
    prepare_session,
    session_progress,
    warn_if_outside_data_root,
)


PROFILES = {
    "base": {"serial_number": "base", "resolution": [4, 2], "fps": 30},
    "wrist": {"serial_number": "wrist", "resolution": [5, 2], "fps": 30},
}
TOPICS = {"operator_twist": "/example_twist"}


def observation():
    return {
        "state": {"arm_joint_position": np.zeros(7, dtype=np.float32)},
        "frames": {
            "base": np.ones((2, 4, 3), dtype=np.uint8),
            "wrist": np.ones((2, 5, 3), dtype=np.uint8),
        },
        "metadata": {"policy_ready": False},
    }


def run_capture(tmp_path, *, stop_after=2, drops=0, max_samples=4):
    ticks = iter(range(stop_after + 1))

    def stop():
        return next(ticks) == stop_after

    return capture_episode(
        tmp_path / "episode_0001", task="peg insertion", fps=10,
        max_samples=max_samples, camera_profiles=PROFILES, topic_names=TOPICS,
        observation=observation,
        drain_events=lambda: ([], drops), stop_requested=stop, sleep=lambda _: None,
    )


def test_manual_stop_is_variable_length_and_preserves_raw_schema(tmp_path):
    report = run_capture(tmp_path)
    manifest = json.loads((tmp_path / "episode_0001/manifest.json").read_text())
    assert report["capture_status"] == "raw_recorded"
    assert report["stop_reason"] == "operator_stop"
    assert report["frames"] == 2
    assert manifest["frames_saved"] == 2
    assert manifest["realworld_episode_success"] is None
    assert manifest["action_targets_7d_available"] is False
    assert report["capture_end_workstation_monotonic_ns"] >= report[
        "capture_start_workstation_monotonic_ns"
    ]
    assert (tmp_path / "episode_0001/frames/base/000001.npy").is_file()


def test_duration_cap_and_dropped_events_do_not_count_as_complete(tmp_path):
    report = run_capture(tmp_path, stop_after=10, max_samples=3)
    assert report["capture_status"] == "incomplete"
    assert report["stop_reason"] == "max_duration"
    report2 = capture_episode(
        tmp_path / "episode_0002", task="peg insertion", fps=10, max_samples=3,
        camera_profiles=PROFILES, topic_names=TOPICS,
        observation=observation,
        drain_events=lambda: ([], 1), stop_requested=lambda: True,
        sleep=lambda _: None,
    )
    assert report2["capture_status"] == "incomplete"


def test_session_resume_counts_reviewed_successes_only_and_never_reuses_index(tmp_path):
    root = tmp_path / "recordings"
    settings = {"schema": "fr3_manual_episode_session_v1", "task": "peg insertion"}
    prepare_session(root, settings)
    prepare_session(root, settings)
    with pytest.raises(ValueError, match="different"):
        prepare_session(root, {**settings, "task": "other"})
    for index, outcome, status in [
        (1, "success", "raw_recorded"),
        (2, "failure", "raw_recorded"),
        (3, "success", "incomplete"),
    ]:
        folder = root / f"episode_{index:04d}"
        folder.mkdir()
        (folder / "episode_review.json").write_text(json.dumps({
            "schema": REVIEW_SCHEMA, "operator_outcome": outcome,
            "capture_status": status,
        }))
    (root / "episode_0004").mkdir()  # Interrupted before operator review.
    assert session_progress(root) == (5, 1)


def test_existing_episode_is_never_overwritten(tmp_path):
    run_capture(tmp_path)
    with pytest.raises(FileExistsError):
        run_capture(tmp_path)


def test_control_c_preserves_interrupted_attempt(tmp_path):
    def interrupt():
        raise KeyboardInterrupt

    report = capture_episode(
        tmp_path / "episode_0001", task="peg insertion", fps=10, max_samples=3,
        camera_profiles=PROFILES, topic_names=TOPICS,
        observation=observation,
        drain_events=lambda: ([], 0), stop_requested=interrupt,
        sleep=lambda _: None,
    )
    assert report["capture_status"] == "interrupted"
    assert report["stop_reason"] == "keyboard_interrupt"
    manifest = json.loads((tmp_path / "episode_0001/manifest.json").read_text())
    assert manifest["status"] == "interrupted"

def test_storage_warning_is_nonblocking_and_checks_resolved_paths(tmp_path, capsys):
    data_root = tmp_path / "data1"
    data_root.mkdir()
    assert not warn_if_outside_data_root(data_root / "fr3_peg_100", data_root)
    assert capsys.readouterr().err == ""
    outside = tmp_path / "home" / "fr3_peg_100"
    assert warn_if_outside_data_root(outside, data_root)
    assert "WARNING" in capsys.readouterr().err


def test_abort_discards_only_current_episode_and_allows_reuse(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    result = capture_episode(
        root / "episode_0007", task="peg insertion", fps=10,
        max_samples=3, camera_profiles=PROFILES, topic_names=TOPICS,
        observation=observation, drain_events=lambda: ([], 0),
        stop_requested=lambda: True, sleep=lambda _: None,
    )
    assert result["capture_status"] == "incomplete"
    discard_aborted_episode(root / "episode_0007", root)
    assert not (root / "episode_0007").exists()
    assert session_progress(root) == (1, 0)
    # A discarded index can be recorded again without touching other episodes.
    run_capture(root)
    assert (root / "episode_0001/manifest.json").is_file()


def test_abort_refuses_other_paths_and_preserves_successful_episode(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    run_capture(root)
    with pytest.raises(ValueError, match="invalid episode path"):
        discard_aborted_episode(root / "episode_0001", tmp_path)
    assert (root / "episode_0001/manifest.json").is_file()
    with pytest.raises(ValueError, match="invalid episode path"):
        discard_aborted_episode(root, root)
