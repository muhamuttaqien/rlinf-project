# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[2]/"scripts"/"prepare_stackcube_openvlaoft_sft.py"
spec=importlib.util.spec_from_file_location("stackcube_sft",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_split_and_alignment(tmp_path):
    src=tmp_path/"source"
    src.mkdir()
    entries=[]
    for i in range(10):
        frames=np.zeros((12,256,256,3),np.uint8)
        frames[:,0,0,0]=np.arange(12)
        actions=np.arange(11*7,dtype=np.float32).reshape(11,7)
        path=f"traj_{i}.npz"
        np.savez(src/path,base_rgb=frames,wrist_rgb=frames,actions=actions)
        entries.append({"episode":f"traj_{i}","path":path})
    (src/"manifest.json").write_text(json.dumps({"episodes":entries}))
    result=module.prepare(src,tmp_path/"out")
    assert len(result["episodes"]["train"])==9
    assert len(result["episodes"]["val"])==1
    assert not set(result["episodes"]["train"]) & set(result["episodes"]["val"])
    example=module.read_example(tmp_path/"out"/"index.json","train",0)
    assert example["base_rgb"].shape==(256,256,3)
    assert example["wrist_rgb"].shape==(256,256,3)
    assert example["actions"].shape==(8,7)
    np.testing.assert_array_equal(example["actions"][0],np.arange(7))
    stats=json.loads((tmp_path/"out"/"train_action_stats.json").read_text())
    assert len(stats["action"]["mean"])==7

def test_invalid_fraction(tmp_path):
    with pytest.raises(ValueError,match="val_fraction"):
        module.prepare(tmp_path,tmp_path/"out",val_fraction=0)
