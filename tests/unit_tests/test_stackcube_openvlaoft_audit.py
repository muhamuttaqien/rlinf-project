# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
import importlib.util
import json
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "scripts" / "audit_stackcube_openvlaoft_sft.py"
spec = importlib.util.spec_from_file_location("audit_sft", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_audit_reports_missing_training_support(tmp_path):
    index = {"format": "stackcube_openvlaoft_sft_intermediate_v1",
             "num_action_chunks": 8, "action_dim": 7,
             "control_mode": "pd_ee_delta_pose",
             "episodes": {"train": ["traj_0"], "val": ["traj_1"]},
             "splits": {"train": [{"step": 0}], "val": [{"step": 0}]}}
    stats = {"action": {key: [1.0]*7 for key in
             ("mean", "std", "q01", "q99", "min", "max")}}
    i, s = tmp_path/"index.json", tmp_path/"stats.json"
    i.write_text(json.dumps(index))
    s.write_text(json.dumps(stats))
    report = module.audit(i, s)
    assert report["train_episodes"] == 1
    assert report["rlinf_generic_sft_worker_supports_openvla_oft"] is False
    assert report["policy_training_started"] is False
