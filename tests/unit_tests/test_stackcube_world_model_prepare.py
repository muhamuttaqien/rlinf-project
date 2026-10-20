# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""CPU tests for StackCube world-model initialization data."""

import importlib.util
import json
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[2] / "scripts" / "prepare_stackcube_world_model.py"
spec = importlib.util.spec_from_file_location("prepare_stackcube_wm", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prepare_reset_data_and_stats(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, :] = np.arange(4, dtype=np.float32)[:, None] / 10
    np.savez(source / "traj_0.npz",
             base_rgb=np.zeros((5, 256, 256, 3), dtype=np.uint8),
             wrist_rgb=np.zeros((5, 256, 256, 3), dtype=np.uint8),
             actions=actions)
    (source / "manifest.json").write_text(
        json.dumps({"episodes": [{"episode": "traj_0", "path": "traj_0.npz"}]})
    )
    output = tmp_path / "output"
    assert module.prepare(source, output) == 1
    episode = np.load(output / "initial_frames" / "traj_0.npy", allow_pickle=True)
    assert episode[0]["image"].shape == (256, 256, 3)
    assert episode[0]["wrist_rgb"].shape == (256, 256, 3)
    assert episode[0]["delta_action"].shape == (7,)
    stats = json.loads((output / "action_stats.json").read_text())
    assert len(stats["action"]["q01"]) == 7
    assert np.all(np.asarray(stats["action"]["q99"]) > stats["action"]["q01"])
