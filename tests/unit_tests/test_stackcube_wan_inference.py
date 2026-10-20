# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""CPU-only preflight tests for Wan's StackCube inference inputs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[2]/"scripts"/"infer_stackcube_wan.py"
spec=importlib.util.spec_from_file_location("wan_infer",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_load_sample_and_checkpoints(tmp_path):
    source=tmp_path/"data"
    source.mkdir()
    frames=np.zeros((14,256,256,3),dtype=np.uint8)
    frames[:,0,0,0]=np.arange(14)
    np.savez(source/"traj_0.npz",base_rgb=frames,
             actions=np.zeros((13,7),dtype=np.float32))
    index=tmp_path/"index.json"
    index.write_text(json.dumps({"backend":"wan","source_dir":str(source),
       "windows":[{"file":"traj_0.npz","action_start":4}]}))
    condition,target,actions,item=module.load_sample(index,0)
    assert condition.shape==(5,256,256,3)
    assert target.shape==(8,256,256,3)
    assert condition[-1,0,0,0]==4
    assert target[0,0,0,0]==5
    assert actions.shape==(8,7)
    ckpt=tmp_path/"weights"
    ckpt.mkdir()
    with pytest.raises(FileNotFoundError):
        module.validate_checkpoint(ckpt)
