# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline-only two-camera FR3 transforms for trajectory-derived proxy BC.

The observed-motion actions are NOT verified controller targets. This module
does not publish ROS topics, call a robot, or approve policy deployment.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _image_hwc(value: object, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must have 3 dimensions")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must be CHW or HWC RGB")
    if np.issubdtype(image.dtype, np.floating):
        # LeRobot returns normalized float32 images in [0, 1], often CHW.
        # Convert explicitly to uint8 without silently accepting bad scales.
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"{name} float RGB values must be finite and in [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 or normalized float RGB")
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class Fr3PegProxyInputs(transforms.DataTransformFn):
    """Map 19D state and two RGB cameras to PI0.5, preserving 7D proxy actions."""

    action_dim: int = 32
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        if self.model_type != _model.ModelType.PI05:
            raise ValueError("This experimental transform supports PI0.5 only")
        if self.action_dim < 19:
            raise ValueError("Model action_dim must accommodate all 19 state values")

        base = _image_hwc(data["observation/image"], "base image")
        wrist = _image_hwc(data["observation/wrist_image"], "wrist image")
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (19,) or not np.isfinite(state).all():
            raise ValueError("Expected 19 finite FR3 state values")

        keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        result = {
            "state": transforms.pad_to_dim(state, self.action_dim),
            "image": dict(zip(keys, (base, wrist, np.zeros_like(base)), strict=True)),
            "image_mask": dict(zip(keys, (np.True_, np.True_, np.False_), strict=True)),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1:] != (7,) or not np.isfinite(actions).all():
                raise ValueError("Expected finite 7D proxy action(s)")
            result["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            result["prompt"] = prompt
        return result


@dataclasses.dataclass(frozen=True)
class Fr3PegProxyOutputs(transforms.DataTransformFn):
    """Slice offline model output to the seven proxy dimensions (NOT robot commands)."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 7:
            raise ValueError("Expected batched padded action predictions")
        return {"actions": actions[:, :7]}
