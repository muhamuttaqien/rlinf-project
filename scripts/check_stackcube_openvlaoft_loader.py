# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Smoke-test the upstream OpenVLA-OFT RLDS data pipeline, without model weights."""
import argparse
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openvla-oft",type=Path,default=Path("external/openvla-oft"))
    p.add_argument("--data-root",type=Path,default=Path("data/stackcube_rlds"))
    args=p.parse_args()
    sys.path.insert(0,str(args.openvla_oft.resolve()))
    from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs
    from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds, apply_trajectory_transforms
    kwargs=make_oxe_dataset_kwargs(
        "stack_cube_builder", args.data_root.resolve(),
        load_camera_views=("primary","wrist"),
        load_proprio=False,load_language=True)
    dataset, stats=make_dataset_from_rlds(
        **kwargs, train=True, shuffle=False, num_parallel_reads=1,
        num_parallel_calls=1)
    # The model receives eight action targets per observation, not a full trajectory.\n    chunked = apply_trajectory_transforms(\n        dataset, train=True, window_size=1, future_action_window_size=7,\n        skip_unlabeled=True, num_parallel_calls=1)\n    element=next(iter(chunked.flatten(num_parallel_calls=1).take(1)))
    print("StackCube OpenVLA-OFT RLDS sample loaded")
    print("Observation keys:",list(element["observation"].keys()))
    print("Action shape:",element["action"].shape)\n    assert element["action"].shape == (8, 7), "Expected one eight-action policy chunk"\n    assert "language_instruction" in element["task"]\n    print("Language:", element["task"]["language_instruction"].numpy().decode())
    print("Action statistics keys:",list(stats["action"].keys()))


if __name__=="__main__":
    main()
