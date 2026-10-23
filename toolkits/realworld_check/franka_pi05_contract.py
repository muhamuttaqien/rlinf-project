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

"""Preflight validation for a real-world OpenPI pi0.5 transform-stage sample."""

from collections.abc import Mapping

import numpy as np


def validate_realworld_pi05_sample(sample: Mapping[str, object]) -> None:
    """Validate shapes and values expected by RLinf's RealworldInputs.

    Validates a temporally windowed transform-stage sample, not the raw
    single-frame LeRobot storage format or the physical action semantics.
    """
    for name in ("observation/image", "observation/state", "actions", "prompt"):
        if name not in sample:
            raise ValueError(f"Missing required OpenPI input: {name}")

    state = np.asarray(sample["observation/state"])
    if state.shape != (19,) or not np.issubdtype(state.dtype, np.number):
        raise ValueError(f"observation/state must be numeric (19,), got {state.shape}, {state.dtype}")
    if not np.all(np.isfinite(state)):
        raise ValueError("observation/state contains nonfinite values")

    action = np.asarray(sample["actions"])
    if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != 7 or not np.issubdtype(action.dtype, np.number):
        raise ValueError(f"actions must be numeric (N, 7), N >= 1; got {action.shape}, {action.dtype}")
    if not np.all(np.isfinite(action)):
        raise ValueError("actions contains nonfinite values")

    for name in ("observation/image", "observation/extra_view_image"):
        if name not in sample:
            if name == "observation/extra_view_image":
                continue  # Upstream RealworldInputs generates a blank second image.
            raise ValueError(f"Missing image: {name}")
        frame = np.asarray(sample[name])
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0 or frame.dtype != np.uint8:
            raise ValueError(f"{name} must be HWC uint8 RGB with 3 channels, got {frame.shape}, {frame.dtype}")

    if not isinstance(sample["prompt"], (str, bytes)) or not sample["prompt"]:
        raise ValueError("prompt must be nonempty text")
