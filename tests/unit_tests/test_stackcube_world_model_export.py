# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""CPU-only tests for the StackCube dataset exporter."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_stackcube_world_model.py"
spec = importlib.util.spec_from_file_location("stackcube_export", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _record(path, *, frames=4, steps=3, action_dim=7):
    with h5py.File(path, "w") as file:
        group = file.create_group("traj_0")
        group.create_dataset("actions", data=np.zeros((steps, 1, action_dim), dtype=np.float32))
        for camera in ("base_camera", "hand_camera"):
            group.create_dataset(
                f"obs/sensor_data/{camera}/rgb",
                data=np.zeros((frames, 1, 8, 8, 3), dtype=np.uint8),
            )


def test_export_valid(tmp_path):
    source = tmp_path / "trajectories.h5"
    _record(source)
    manifest = module.convert(source, tmp_path / "converted")
    assert manifest["episodes"][0]["steps"] == 3
    result = np.load(tmp_path / "converted" / "traj_0.npz")
    assert result["rgb"].shape == (4, 256, 256, 3)
    assert result["actions"].shape == (3, 7)


def test_rejects_misaligned_frames(tmp_path):
    source = tmp_path / "trajectories.h5"
    _record(source, frames=3)
    with pytest.raises(ValueError, match="T\\+1 RGB"):
        module.convert(source, tmp_path / "converted")


def test_rejects_wrong_action_dimension(tmp_path):
    source = tmp_path / "trajectories.h5"
    _record(source, action_dim=8)
    with pytest.raises(ValueError, match="7-D Panda"):
        module.convert(source, tmp_path / "converted")


def test_rejects_invalid_image_size(tmp_path):
    source = tmp_path / "trajectories.h5"
    _record(source)
    with pytest.raises(ValueError, match="image_size must be positive"):
        module.convert(source, tmp_path / "converted", image_size=0)


def test_export_both_cameras(tmp_path):
    source = tmp_path / "trajectories.h5"
    _record(source)
    manifest = module.convert(source, tmp_path / "converted",
                              cameras=("base_camera", "hand_camera"))
    assert manifest["cameras"] == ["base_camera", "hand_camera"]
    with np.load(tmp_path / "converted" / "traj_0.npz") as data:
        assert data["base_rgb"].shape == (4, 256, 256, 3)
        assert data["wrist_rgb"].shape == (4, 256, 256, 3)
        assert data["actions"].shape == (3, 7)


def test_rejects_missing_second_camera(tmp_path):
    source = tmp_path / "trajectories.h5"
    _record(source)
    with h5py.File(source, "r+") as file:
        del file["traj_0/obs/sensor_data/hand_camera"]
    with pytest.raises(ValueError, match="No RGB observation"):
        module.convert(source, tmp_path / "converted",
                       cameras=("base_camera", "hand_camera"))
