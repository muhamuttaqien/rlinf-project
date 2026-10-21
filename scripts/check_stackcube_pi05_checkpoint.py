# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check RLinf Pi0.5 checkpoint tensor shapes without loading model weights.

Constructs the exact StackCube Pi0.5 model architecture on the PyTorch meta
device, compares every state_dict name and shape against safetensors headers,
and avoids allocating a full model or any CUDA memory. Does not perform
training or validate numerical quality of checkpoint weights.

Run from the RLinf repository root:
    PYTHONPATH="$PWD" python scripts/check_stackcube_pi05_checkpoint.py \\
      --checkpoint checkpoints/torch/pi05_base_rlinf/model.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/torch/pi05_base_rlinf/model.safetensors"),
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    official = get_openpi_config("pi05_stackcube_panda_joint").model
    assert official.pi05 is True
    assert official.action_horizon == 10
    assert official.action_dim == 32
    assert official.discrete_state_input is False

    model_cfg = Pi0Config(
        pi05=official.pi05,
        action_horizon=official.action_horizon,
        action_dim=official.action_dim,
        max_token_len=official.max_token_len,
        discrete_state_input=official.discrete_state_input,
        paligemma_variant=official.paligemma_variant,
        action_expert_variant=official.action_expert_variant,
        dtype="bfloat16",
        pcd=False,
    )
    set_torch_compile(False)
    with torch.device("meta"):
        model = Pi0(
            model_cfg,
            num_steps=5,
            action_env_dim=8,
            action_chunk=10,
        )
    expected = {
        name: tuple(tensor.shape)
        for name, tensor in model.state_dict().items()
    }

    with safe_open(checkpoint, framework="pt", device="cpu") as reader:
        checkpoint_names = set(reader.keys())
        expected_names = set(expected)
        missing = sorted(expected_names - checkpoint_names)
        unexpected = sorted(checkpoint_names - expected_names)
        mismatches = sorted(
            (
                name,
                expected[name],
                tuple(reader.get_slice(name).get_shape()),
            )
            for name in checkpoint_names & expected_names
            if tuple(reader.get_slice(name).get_shape()) != expected[name]
        )

    print(f"Checkpoint: {checkpoint}")
    print(f"Model parameters (approx): {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B")
    print(f"Expected state tensors: {len(expected)}")
    print(f"Checkpoint tensors: {len(checkpoint_names)}")
    print(f"Missing model tensors: {len(missing)}")
    print(f"Unexpected checkpoint tensors: {len(unexpected)}")
    print(f"Mismatched tensor shapes: {len(mismatches)}")
    for category, entries in (
        ("Missing", missing),
        ("Unexpected", unexpected),
        ("Shape mismatches", mismatches),
    ):
        for entry in entries[:12]:
            print(f"  {category}: {entry}")

    if missing or unexpected or mismatches:
        raise RuntimeError(
            "Checkpoint tensor metadata does not match the StackCube Pi0.5 "
            "model; do not launch SFT until this is resolved."
        )
    print("StackCube Pi0.5 checkpoint architecture: PASSED")
    print("No checkpoint weights or CUDA tensors were materialized.")


if __name__ == "__main__":
    main()
