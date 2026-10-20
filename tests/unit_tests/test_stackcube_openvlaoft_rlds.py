# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""CPU-only schema checks; TFDS integration requires optional dependencies."""
import importlib.util
import json
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parents[2]/"scripts"/"export_stackcube_openvlaoft_rlds.py"
spec=importlib.util.spec_from_file_location("stackcube_rlds",p)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rlds_step_alignment(tmp_path):
    source=tmp_path/"source"
    source.mkdir()
    imgs=np.zeros((4,256,256,3),np.uint8)
    actions=np.arange(21,dtype=np.float32).reshape(3,7)
    np.savez(source/"traj_0.npz",base_rgb=imgs,wrist_rgb=imgs,actions=actions)
    steps=list(module.episode_steps(source,{"path":"traj_0.npz"},"Stack cubes"))
    assert len(steps)==3
    np.testing.assert_array_equal(steps[1]["action"],actions[1])
    assert steps[0]["is_first"] is True
    assert steps[-1]["is_last"] is True
    assert steps[0]["observation"]["wrist_image"].shape==(256,256,3)


def test_tfds_info_does_not_use_config_discovery():
    # Standalone TFDS builders must not search for dataset metadata adjacent
    # to this script's .py file.
    exporter = Path(module.__file__).read_text()
    assert "dataset_info_from_configs" not in exporter
    assert "tfds.core.DatasetInfo(" in exporter
