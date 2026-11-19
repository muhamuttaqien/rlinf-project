# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Record read-only FR3 pi0.5 shadow predictions; never label them as rollouts.

The existing official SFT validation loader supplies the two-camera observations,
19D state, and source action labels to the policy. This recorder references those
immutable inputs rather than duplicating their large image/video files.
There is no ROS import, hardware control, or success/failure fabrication.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np


def _request_chunk(values, *, label):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (10, 7) or not np.isfinite(arr).all():
        raise ValueError(f"{label} must be a finite (10,7) request array")
    return arr.tolist()


class ShadowRolloutRecorder:
    """New directory with a JSONL action trace and a manifest, NO robot actions."""

    def __init__(self, directory: Path, *, checkpoint: Path, validation: Path):
        self.directory = Path(directory).expanduser()
        # Fail rather than overwrite any previous trial.
        self.directory.mkdir(parents=True, exist_ok=False)
        self._stream = (self.directory / "shadow_predictions.jsonl").open(
            "x", encoding="utf-8"
        )
        self.checkpoint = str(Path(checkpoint).resolve())
        self.validation = str(Path(validation).resolve())
        self.count = 0
        self.episodes = set()

    def record(self, *, window_index: int, source_episode: str,
               prediction, reference) -> None:
        if (type(window_index) is not int or window_index < 0
                or not re.fullmatch(r"episode_[0-9]{4,}", source_episode)):
            raise ValueError("Invalid validation source/index")
        pred = _request_chunk(prediction, label="prediction")
        truth = _request_chunk(reference, label="reference")
        if (np.max(np.abs(np.asarray(truth)[:, :6])) > 1.002
                or not np.isin(np.round(np.asarray(truth)[:, 6]), [0, 1]).all()):
            raise ValueError("Reference labels are not normalized SpaceMouse requests")
        row = {
            "source_episode": source_episode,
            "source_strict_window_index": window_index,
            "observation_source": {
                "dataset_root": self.validation,
                "index": window_index,
                "images_and_state": "referenced_in_existing_LeRobot_dataset",
            },
            "predicted_request_chunk_7d": pred,
            "recorded_operator_request_chunk_7d": truth,
            "controller_accepted_action": None,
            "executed_action": None,
            "is_success": None,
            "mode": "offline_shadow_only",
            "robot_commands_sent": 0,
        }
        self._stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.count += 1
        self.episodes.add(source_episode)

    def close(self) -> Path:
        if self._stream.closed:
            raise RuntimeError("Recorder already closed")
        self._stream.close()
        report = {
            "schema": "fr3_pi05_request_offline_shadow_v1",
            "mode": "offline_shadow_only",
            "checkpoint": self.checkpoint,
            "validation_dataset": self.validation,
            "predicted_chunks": self.count,
            "source_episodes": len(self.episodes),
            "actions_file": "shadow_predictions.jsonl",
            "observation_storage": "references_only_existing_validation_LeRobot",
            "robot_commands_sent": 0,
            "success_failure_labels_collected": 0,
            "physical_execution_approved": False,
            "recap_rollout_training_ready": False,
            "note": (
                "Teacher-forced SFT validation replay, not a real policy rollout. "
                "Robot outcome labels and executed-controller actions are unavailable."
            ),
        }
        path = self.directory / "manifest.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return path
