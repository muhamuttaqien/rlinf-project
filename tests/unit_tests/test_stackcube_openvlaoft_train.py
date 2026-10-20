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

def test_qlora_training_command():
    cmd = module.training_command(
        Path("external/openvla-oft"), Path("/tmp/stackcube_rlds"),
        "openvla/openvla-7b", Path("/tmp/output"), 1, 4, qlora=True
    )
    assert cmd[cmd.index("--use_qlora") + 1] == "True"
    assert cmd[cmd.index("--gradient_checkpointing") + 1] == "True"
    assert cmd[cmd.index("--num_images_in_input") + 1] == "1"


def test_upstream_qlora_patch_idempotent():
    path = Path(__file__).resolve().parents[2] / "scripts" / "patch_stackcube_openvlaoft_qlora.py"
    compile(path.read_text(), str(path), "exec")
    import importlib.util
    spec = importlib.util.spec_from_file_location("qlora_patch", path)
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    sample = (
        'from peft import LoraConfig, PeftModel, get_peft_model\n'
        'from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor\n'
        '    lora_dropout: float = 0.0\n'
        + patcher.ORIGINAL_LOAD + '\n'
        '            target_modules="all-linear",\n'
        '            if log_step == cfg.max_steps:\n'
        '            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:\n'
    )
    first, changed = patcher.patch_source(sample)
    assert changed
    assert "BitsAndBytesConfig(" in first
    assert "gradient_checkpointing_enable()" in first
    second, changed_again = patcher.patch_source(first)
    assert not changed_again and second == first
