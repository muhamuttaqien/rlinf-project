# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Evidence-based 20-episode outcome registration, without robot access."""

import json

import pytest

from toolkits.realworld_check.fr3_rollout_outcomes import progress, validate_episode


def sample(tmp_path, *, outcome="success", commands=1, shadow=False):
    (tmp_path / "command.jsonl").write_text('{"action":[0,0,0,0,0,0,1]}\n')
    (tmp_path / "state.jsonl").write_text('{"tcp":[0,0,0]}\n')
    return {
        "episode_id": 1,
        "policy_checkpoint": "/checkpoints/global_step_3000/actor/full_weights.pt",
        "policy_command_log": "command.jsonl",
        "measured_state_log": "state.jsonl",
        "outcome": outcome,
        "operator_reviewed": True,
        "robot_commands_sent": commands,
        "shadow_only": shadow,
    }


def test_valid_real_episode(tmp_path):
    validate_episode(sample(tmp_path), root=tmp_path)


@pytest.mark.parametrize("commands,shadow", [(0, False), (5, True)])
def test_rejects_shadow_or_no_actuation(tmp_path, commands, shadow):
    with pytest.raises(ValueError):
        validate_episode(
            sample(tmp_path, commands=commands, shadow=shadow), root=tmp_path
        )


def test_progress_requires_labeled_physical_evidence(tmp_path):
    folder = tmp_path / "episode_01"
    folder.mkdir()
    item = sample(folder, outcome="failure")
    (folder / "outcome.json").write_text(json.dumps(item))
    status = progress(tmp_path)
    assert status["real_rollouts"] == 1
    assert status["failures"] == 1
    assert status["next_episode"] == 2
    assert status["recap_dataset_ready"] is False


def test_aborted_attempt_not_counted_as_usable_rollout(tmp_path):
    folder = tmp_path / "episode_01"
    folder.mkdir()
    item = sample(folder, outcome="aborted")
    (folder / "outcome.json").write_text(json.dumps(item))
    status = progress(tmp_path)
    assert status["recorded_attempts"] == 1
    assert status["real_rollouts"] == 0
    assert status["aborted"] == 1
    assert status["next_episode"] == 2


def test_incomplete_episode_preserved_and_skipped(tmp_path):
    folder = tmp_path / "episode_01"
    folder.mkdir()
    (folder / "policy_command_log.jsonl").write_text('{"action": [0]}\n')
    status = progress(tmp_path)
    assert status["incomplete_attempts"] == 1
    assert status["real_rollouts"] == 0
    assert status["next_episode"] == 2


def test_aborted_attempts_do_not_exhaust_target_20(tmp_path):
    for i in range(1, 21):
        folder = tmp_path / f"episode_{i:02d}"
        folder.mkdir()
        item = sample(folder, outcome="aborted")
        item["episode_id"] = i
        (folder / "outcome.json").write_text(json.dumps(item))
    status = progress(tmp_path)
    assert status["recorded_attempts"] == 20
    assert status["real_rollouts"] == 0
    assert status["next_episode"] == 21


def test_100_plus_attempt_id_is_discoverable(tmp_path):
    folder = tmp_path / "episode_101"
    folder.mkdir()
    item = sample(folder, outcome="aborted")
    item["episode_id"] = 101
    (folder / "outcome.json").write_text(json.dumps(item))
    result = progress(tmp_path)
    assert result["recorded_attempts"] == 1
    assert result["next_episode"] == 1


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_new_results_require_interface_qualification_to_count(
    tmp_path, qualified, outcome
):
    folder = tmp_path / "episode_01"
    folder.mkdir()
    item = sample(folder, outcome=outcome)
    item["execution_interface_qualified"] = qualified
    (folder / "outcome.json").write_text(json.dumps(item))
    status = progress(tmp_path)
    assert status["recorded_attempts"] == 1
    assert status["real_rollouts"] == int(qualified)
    assert status["next_episode"] == 2
    assert json.loads((folder / "outcome.json").read_text()) == item


@pytest.mark.parametrize("commands", [0, 6])
def test_unreviewed_abort_preserved_without_counting(tmp_path, commands):
    folder = tmp_path / "episode_08"
    folder.mkdir()
    item = sample(folder, outcome="aborted", commands=commands)
    item.update(episode_id=8, operator_reviewed=False, shutdown_confirmed=False)
    if not commands:
        (folder / "command.jsonl").write_text("")
        (folder / "state.jsonl").write_text("")
    (folder / "outcome.json").write_text(json.dumps(item))
    status = progress(tmp_path)
    assert status["recorded_attempts"] == 1
    assert status["aborted"] == 1
    assert status["real_rollouts"] == 0


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_unreviewed_result_cannot_count(tmp_path, outcome):
    item = sample(tmp_path, outcome=outcome)
    item.update(operator_reviewed=False, execution_interface_qualified=True)
    with pytest.raises(ValueError, match="operator-reviewed"):
        validate_episode(item, root=tmp_path)


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_qualification_flag_requires_a_boolean(tmp_path, value):
    item = sample(tmp_path)
    item["execution_interface_qualified"] = value
    with pytest.raises(ValueError, match="boolean"):
        validate_episode(item, root=tmp_path)
