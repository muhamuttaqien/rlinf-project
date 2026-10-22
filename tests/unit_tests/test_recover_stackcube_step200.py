# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only recovery tests using tiny, real PyTorch DCP shards."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from scripts.recover_stackcube_step200 import run


def make_example(tmp_path):
    root = tmp_path / "results"
    for step in (100, 200):
        actor = root / "checkpoints" / f"global_step_{step}" / "actor"
        directory = actor / "dcp_checkpoint"
        directory.mkdir(parents=True)
        state = {
            "fsdp_checkpoint": {
                "model": {
                    "llm.weight": torch.full((4, 3), float(step)),
                    "img.bias": torch.full((5,), float(step) / 100),
                },
                "optimizers": {"state": {"moment": torch.zeros(10)}},
                "fsdp_version": "fsdp",
            }
        }
        dcp.save(state, checkpoint_id=str(directory))
        if step == 100:
            full = actor / "model_state_dict" / "full_weights.pt"
            full.parent.mkdir()
            torch.save(state["fsdp_checkpoint"]["model"], full)
        else:
            (directory / ".metadata").unlink()
    return root


def options(root, tmp_path, *, recover=False, use_template=False):
    return SimpleNamespace(
        results_root=root,
        output=tmp_path / "recovery/actor/model_state_dict/full_weights.pt",
        recover=recover,
        use_step100_layout=use_template,
    )


def test_recover_model_only_from_step200_shard_using_step100_layout(tmp_path):
    root = make_example(tmp_path)
    args = options(root, tmp_path)
    run(args)
    assert not args.output.exists()

    args.recover = True
    args.use_step100_layout = True
    run(args)
    weights = torch.load(args.output, map_location="cpu", weights_only=True)
    assert set(weights) == {"llm.weight", "img.bias"}
    assert weights["llm.weight"][0, 0].item() == 200
    assert (
        root / "checkpoints/global_step_200/actor/dcp_checkpoint/__0_0.distcp"
    ).is_file()
    assert not (
        root / "checkpoints/global_step_200/actor/dcp_checkpoint/.metadata"
    ).exists()


def test_refuse_truncated_shard_before_writing(tmp_path):
    root = make_example(tmp_path)
    shard = root / "checkpoints/global_step_200/actor/dcp_checkpoint/__0_0.distcp"
    with shard.open("r+b") as stream:
        stream.truncate(100)
    args = options(root, tmp_path)
    with pytest.raises(ValueError, match="Incomplete shard"):
        run(args)
    assert not args.output.exists()


def test_missing_step200_metadata_needs_explicit_recovery_opt_in(tmp_path):
    root = make_example(tmp_path)
    args = options(root, tmp_path, recover=True)
    with pytest.raises(ValueError, match="--use-step100-layout"):
        run(args)
    assert not args.output.exists()
