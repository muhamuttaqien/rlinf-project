# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
import importlib.util
import json
from pathlib import Path
import pytest

p = Path(__file__).resolve().parents[2] / "scripts" / "train_stackcube_openvlaoft.py"
spec = importlib.util.spec_from_file_location("stackcube_train", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_launch_command_and_guard(tmp_path):
    root = tmp_path / "openvla-oft"
    (root / "vla-scripts").mkdir(parents=True)
    (root / "vla-scripts" / "finetune.py").touch()
    reg = root / "prismatic/vla/datasets/rlds/oxe/configs.py"
    reg.parent.mkdir(parents=True)
    reg.write_text('"stack_cube_builder": {}')
    data = tmp_path / "rlds"
    dataset = data / "stack_cube_builder" / "1.0.0"
    dataset.mkdir(parents=True)
    (dataset / "dataset_info.json").write_text("{}")
    assert module.validate_paths(root, data) == (root.resolve(), data.resolve())
    cmd = module.training_command(root, data, "openvla/openvla-7b", tmp_path/"out", 1, 8)
    assert "--dataset_name" in cmd and "stack_cube_builder" in cmd
    assert "--max_steps" in cmd and cmd[cmd.index("--max_steps") + 1] == "1"
    assert "--merge_lora_during_training" in cmd
    with pytest.raises(ValueError):
        module.training_command(root, data, "openvla/openvla-7b", tmp_path/"out", 0, 8)


def test_launcher_python_syntax():
    compile(p.read_text(), str(p), "exec")
