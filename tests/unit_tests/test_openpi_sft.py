# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""OpenPI SFT memory configuration contracts."""

from types import SimpleNamespace

import torch


def test_sft_freezes_vision_language_trunk_when_requested():
    from rlinf.models.embodiment.openpi import _freeze_after_load

    class Model:
        def __init__(self):
            self.frozen = False

        def freeze_vlm(self):
            self.frozen = True
            return 2

    model = Model()
    _freeze_after_load(model, "sft", train_expert_only=True)
    assert model.frozen


def test_sft_forward_preserves_enabled_gradient_checkpointing():
    from rlinf.models.embodiment.openpi.pi0 import Pi0

    class Model:
        rlt_cfg = SimpleNamespace(use_rlt=False)

        def gradient_checkpointing_disable(self):
            raise AssertionError("SFT must preserve configured checkpointing")

        def _unpack_sft_batch(self, data):
            return "observation", "actions"

        def _observation_to_device(self, observation):
            return observation

        def _actions_to_device(self, actions):
            return actions

        def compute_loss(self, observation, actions, train):
            assert train
            return torch.ones(1, 1, 1)

        def _reduce_sft_loss(self, loss, use_action_chunk_loss):
            assert not use_action_chunk_loss
            return loss.mean()

    loss = Pi0.sft_forward(Model(), data={})
    assert loss.item() == 1.0
