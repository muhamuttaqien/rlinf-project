# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""CPU-only window indexing and alignment tests."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

p = Path(__file__).resolve().parents[2] / "scripts" / "index_stackcube_world_model.py"
spec = importlib.util.spec_from_file_location("stackcube_windows", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "episodes"
    path.mkdir()
    actions = np.arange(20 * 7, dtype=np.float32).reshape(20, 7)
    frames = np.zeros((21, 256, 256, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(21)
    np.savez(path / "traj_0.npz", base_rgb=frames, wrist_rgb=frames,
             actions=actions)
    (path / "manifest.json").write_text(json.dumps(
        {"episodes": [{"episode": "traj_0", "path": "traj_0.npz"}]}))
    return path


@pytest.mark.parametrize("backend,condition", [("wan", 5), ("opensora", 4)])
def test_aligned_windows(source, tmp_path, backend, condition):
    dest = tmp_path / f"{backend}.json"
    idx = module.build_index(source, dest, backend)
    assert idx["num_windows"] > 0
    sample = module.read_window(dest, 0)
    assert sample["base_condition"].shape == (condition, 256, 256, 3)
    assert sample["wrist_target"].shape == (8, 256, 256, 3)
    assert sample["actions"].shape == (8, 7)
    assert sample["base_condition"][-1, 0, 0, 0] == condition - 1
    assert sample["base_target"][0, 0, 0, 0] == condition


def test_bad_stride(source, tmp_path):
    with pytest.raises(ValueError, match="stride"):
        module.build_index(source, tmp_path / "x.json", "wan", stride=0)
