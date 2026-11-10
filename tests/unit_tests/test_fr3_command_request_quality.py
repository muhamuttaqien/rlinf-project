"""Offline candidate quality audit tests; no ROS, robot, or SFT."""
import json

import pytest

from toolkits.realworld_check.fr3_command_request_quality import audit


def _candidate(root, episode, grips):
    path = root / "episodes"
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, g in enumerate(grips):
        complete = g is not None
        arm = [0.1] * 6
        rows.append({
            "frame_index": i, "next_frame_index": i+1,
            "observation_dt_s": 0.1,
            "twist_request_time_coverage": 1.0,
            "spacemouse_mean_request_6d": arm,
            "gripper_request_open_fraction": g,
            "candidate_request_action_7d": arm + [g] if complete else None,
            "candidate_complete": complete,
        })
    (path / f"{episode}.jsonl").write_text(
        "".join(json.dumps(x)+"\n" for x in rows), encoding="utf-8"
    )
    return {
        "episode": episode,
        "transitions": len(rows),
        "counts": {"candidate_7d_available": sum(g is not None for g in grips)},
    }


def test_audit_prefix_and_episode_disjoint_split(tmp_path):
    a = _candidate(tmp_path, "episode_0001", [None, None, 1.0, 1.0, 0.0])
    b = _candidate(tmp_path, "episode_0002", [None, 0.0, 0.0, 1.0])
    (tmp_path / "report.json").write_text(json.dumps({
        "schema": "fr3_spacemouse_request_alignment_candidate_v1",
        "episode_reports": [a, b],
        "min_twist_time_coverage": .9,
        "transitions": 9, "candidate_7d_available": 6,
    }))
    report = audit(tmp_path, horizon=2, validation_fraction=.5)
    assert report["counts"]["initial_unknown_gripper_prefix"] == 3
    assert report["counts"]["complete_7d_request"] == 6
    assert sum(x["complete_7d_horizon_windows"]
               for x in report["episode_manifests"]) == 4
    assert report["train"]["episodes"] == 1
    assert report["validation"]["episodes"] == 1
    assert report["sft_ready"] is False


def test_reject_fake_complete_action(tmp_path):
    a = _candidate(tmp_path, "episode_0001", [None, 1.0])
    b = _candidate(tmp_path, "episode_0002", [1.0, 0.0])
    path = tmp_path / "episodes" / "episode_0001.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["candidate_request_action_7d"] = [0.] * 7
    path.write_text("".join(json.dumps(x)+"\n" for x in rows))
    (tmp_path / "report.json").write_text(json.dumps({
        "schema": "fr3_spacemouse_request_alignment_candidate_v1",
        "episode_reports": [a, b],
        "min_twist_time_coverage": .9,
        "transitions": 4, "candidate_7d_available": 3,
    }))
    with pytest.raises(ValueError, match="incomplete row has action"):
        audit(tmp_path)
