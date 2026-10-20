# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Register StackCube in a local checkout of moojink/openvla-oft.

Run this once after cloning the upstream repository. No model is downloaded.
"""
import argparse
from pathlib import Path

DATASET = "stack_cube_builder"
CONFIG = '''    "stack_cube_builder": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
'''
MIXTURE = '''    "stack_cube_builder": [("stack_cube_builder", 1.0)],
'''
TRANSFORM = '''    "stack_cube_builder": lambda trajectory: trajectory,
'''


def insert_registry(path, marker, snippet):
    text = path.read_text()
    if marker in text:
        return False
    pos = text.rfind("\n}")
    if pos < 0:
        raise ValueError(f"Could not find registry closing bracket in {path}")
    path.write_text(text[:pos] + "\n" + snippet + text[pos:])
    return True


def register(root):
    root = Path(root)
    directory = root / "prismatic/vla/datasets/rlds/oxe"
    targets = [
        (directory / "configs.py", '"stack_cube_builder": {', CONFIG),
        (directory / "mixtures.py", '"stack_cube_builder": [', MIXTURE),
        (directory / "transforms.py", '"stack_cube_builder": lambda', TRANSFORM),
    ]
    for path, marker, snippet in targets:
        if not path.is_file():
            raise FileNotFoundError(f"Not an OpenVLA-OFT checkout: missing {path}")
        status = "updated" if insert_registry(path, marker, snippet) else "already registered"
        print(f"{status}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvla-oft", type=Path, required=True)
    args = parser.parse_args()
    register(args.openvla_oft)


if __name__ == "__main__":
    main()
