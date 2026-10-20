# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Reference evaluation of temporally indexed dual-camera sequences."""

import importlib.util
import json
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_stackcube_prediction.py"
spec = importlib.util.spec_from_file_location("stackcube_eval", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_persistence_metrics(tmp_path):
    source = tmp_path / "episodes"
    source.mkdir()
    images = np.zeros((14, 256, 256, 3), dtype=np.uint8)
    images[5:] = 20
    np.savez(source / "traj_0.npz", base_rgb=images, wrist_rgb=images,
             actions=np.zeros((13, 7), dtype=np.float32))
    index = {"source_dir": str(source), "condition_frames": 5, "chunk": 8,
             "windows": [{"episode": "traj_0", "file": "traj_0.npz", "action_start": 4}]}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    result = module.evaluate(path, tmp_path / "results")
    assert result["num_windows"] == 1
    assert result["base_mae_0_255"] == 20
    assert result["wrist_mae_0_255"] == 20
    assert (tmp_path / "results" / "base_persistence_comparison.png").exists()
