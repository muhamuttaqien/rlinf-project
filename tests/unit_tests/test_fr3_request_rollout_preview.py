"""No-hardware checks of SFT request preview and rollout readiness gates."""

import json

import numpy as np
import pytest

from toolkits.realworld_check.fr3_request_rollout_preview import (
    preview_request, summarize_live_shadow,
)


def test_gripper_binary_preview_and_ambiguous_hold():
    assert preview_request([0, 0, 0, 0, 0, 0, -0.02])["gripper_intent"] == "close_request_preview"
    assert preview_request([0, 0, 0, 0, 0, 0, 1.04])["gripper_intent"] == "open_request_preview"
    assert preview_request([0, 0, 0, 0, 0, 0, 0.5])["gripper_intent"] == "hold_ambiguous_gripper_preview"


@pytest.mark.parametrize("request", [
    [1.001, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1.2],
    [0, 0, 0, 0, 0, 0, float("nan")],
    [0, 0, 0],
])
def test_rejects_bad_requests(request):
    with pytest.raises(ValueError):
        preview_request(request)


def test_live_shadow_summary_never_approves_physical_execution(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "mode": "live_sensor_shadow_inference_only",
        "robot_commands_sent": 0,
        "physical_execution_approved": False,
        "samples": 1,
    }))
    (tmp_path / "predictions.jsonl").write_text(json.dumps({
        "sample": 0, "robot_commands_sent": 0, "executed_action": None,
        "is_success": None, "inference_elapsed_s": 0.45,
        "predicted_request_chunk_7d": np.zeros((10, 7)).tolist(),
    }) + "\n")
    result = summarize_live_shadow(tmp_path)
    assert result["total_predicted_steps"] == 10
    assert result["inference_may_outlast_command_timeout"] is True
    assert result["physical_execution_ready"] is False
    assert result["robot_commands_sent"] == 0


def test_rejects_mislabeled_motion_trace(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "mode": "live_sensor_shadow_inference_only",
        "robot_commands_sent": 1,
        "physical_execution_approved": False,
        "samples": 0,
    }))
    (tmp_path / "predictions.jsonl").write_text("")
    with pytest.raises(ValueError):
        summarize_live_shadow(tmp_path)
