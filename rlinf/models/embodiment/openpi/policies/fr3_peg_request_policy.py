# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""OFFLINE-ONLY PI0.5 transforms for normalized FR3 SpaceMouse REQUESTS.

NOT a trajectory-derived pose-delta policy. NOT controller-accepted actions.
NO hardware controller adapter, ROS publisher, or deployment permission.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi import transforms
from openpi.models import model as _model

from rlinf.models.embodiment.openpi.policies.fr3_peg_proxy_policy import _image_hwc


@dataclasses.dataclass(frozen=True)
class Fr3PegRequestInputs(transforms.DataTransformFn):
    """19D observed state, two RGB views, and 7D normalized request labels."""

    action_dim: int = 32
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        if self.model_type != _model.ModelType.PI05 or self.action_dim != 32:
            raise ValueError("FR3 command-request diagnostic requires PI0.5 action_dim=32")
        base = _image_hwc(data["observation/image"], "base image")
        wrist = _image_hwc(data["observation/wrist_image"], "wrist image")
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (19,) or not np.isfinite(state).all():
            raise ValueError("Expected 19 finite FR3 state features")
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        result = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": dict(zip(names, (base, wrist, np.zeros_like(base)), strict=True)),
            "image_mask": dict(zip(
                names, (np.True_, np.True_, np.False_), strict=True
            )),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if (actions.ndim != 2 or actions.shape[-1] != 7
                    or not np.isfinite(actions).all()
                    or np.any(np.abs(actions[:, :6]) > 1.001)
                    or np.any(~np.isin(actions[:, 6], [0.0, 1.0]))):
                raise ValueError(
                    "Expected (horizon,7) bounded normalized SpaceMouse requests"
                )
            result["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            result["prompt"] = prompt
        return result


@dataclasses.dataclass(frozen=True)
class Fr3PegRequestOutputs(transforms.DataTransformFn):
    """Return 7D normalized requests for OFFLINE analysis only, never hardware."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 7:
            raise ValueError("Expected batched padded actions")
        return {"actions": actions[:, :7]}
