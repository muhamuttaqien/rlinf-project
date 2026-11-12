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
        validate_episode(sample(tmp_path, commands=commands, shadow=shadow), root=tmp_path)


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
