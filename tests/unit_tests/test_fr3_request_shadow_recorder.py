"""CPU-only tests for the non-actuating FR3 rollout shadow recorder."""

import json

import numpy as np
import pytest

from toolkits.realworld_check.fr3_request_shadow_recorder import ShadowRolloutRecorder


def test_shadow_writes_predictions_and_never_fabricates_outcomes(tmp_path):
    out = tmp_path / "shadow"
    recorder = ShadowRolloutRecorder(
        out, checkpoint=tmp_path / "model.pt", validation=tmp_path / "dataset"
    )
    prediction = np.zeros((10, 7), dtype=float)
    reference = np.zeros((10, 7), dtype=float)
    reference[:, 6] = 1
    recorder.record(
        window_index=4, source_episode="episode_0002",
        prediction=prediction, reference=reference,
    )
    manifest_path = recorder.close()
    manifest = json.loads(manifest_path.read_text())
    row = json.loads((out / "shadow_predictions.jsonl").read_text())
    assert manifest["predicted_chunks"] == 1
    assert manifest["success_failure_labels_collected"] == 0
    assert manifest["recap_rollout_training_ready"] is False
    assert row["is_success"] is None
    assert row["executed_action"] is None
    assert row["controller_accepted_action"] is None
    assert row["robot_commands_sent"] == 0
    assert row["predicted_request_chunk_7d"] == prediction.tolist()
    with pytest.raises(FileExistsError):
        ShadowRolloutRecorder(
            out, checkpoint=tmp_path / "model.pt", validation=tmp_path / "dataset"
        )


def test_shadow_rejects_invalid_actions(tmp_path):
    recorder = ShadowRolloutRecorder(
        tmp_path / "shadow", checkpoint=tmp_path / "model.pt",
        validation=tmp_path / "dataset",
    )
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros((10, 7))
        bad[0, 0] = np.nan
        recorder.record(
            window_index=0, source_episode="episode_0001",
            prediction=bad, reference=np.zeros((10, 7)),
        )
    with pytest.raises(ValueError, match="normalized"):
        bad = np.zeros((10, 7))
        bad[:, 0] = 2.0
        recorder.record(
            window_index=0, source_episode="episode_0001",
            prediction=np.zeros((10, 7)), reference=bad,
        )
    assert json.loads(recorder.close().read_text())["predicted_chunks"] == 0
