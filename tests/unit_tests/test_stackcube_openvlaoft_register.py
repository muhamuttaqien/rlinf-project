# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
import importlib.util
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "scripts" / "register_stackcube_openvlaoft.py"
spec = importlib.util.spec_from_file_location("stackcube_register", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_registry_idempotent(tmp_path):
    root = tmp_path / "openvla-oft"
    d = root / "prismatic/vla/datasets/rlds/oxe"
    d.mkdir(parents=True)
    for name in ("configs.py", "transforms.py", "mixtures.py"):
        (d / name).write_text("REGISTRY = {\n}\n")
    loader = root / "prismatic/vla/datasets/rlds/dataset.py"
    loader.parent.mkdir(parents=True, exist_ok=True)
    loader.write_text('    builder = tfds.builder(name, data_dir=data_dir)\\n    split = "train" if train else "val"\\n')
    module.register(root)
    first = {x.name: x.read_text() for x in d.iterdir()}
    module.register(root)
    assert first == {x.name: x.read_text() for x in d.iterdir()}
    assert '"stack_cube_builder"' in first["configs.py"]
    assert '"stack_cube_builder"' in first["mixtures.py"]
    assert '"stack_cube_builder"' in first["transforms.py"]
    assert "builder_from_directory" in loader.read_text()
    assert '"validation"' in loader.read_text()
