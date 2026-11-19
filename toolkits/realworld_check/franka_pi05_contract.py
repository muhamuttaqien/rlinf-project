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

"""Preflight validation for a two-camera OpenPI pi0.5 transform-stage sample."""

from collections.abc import Mapping

import numpy as np


def validate_realworld_pi05_sample(sample: Mapping[str, object]) -> None:
    """Validate the candidate two-camera Franka SFT sample contract.

    The upstream RealworldInputs permits a missing second image by injecting
    a blank image, but that is invalid for this two-camera experiment. This
    check does not establish the robot's 19D state mapping or 7D action units.
    """
    required = (
        "observation/image",
        "observation/extra_view_image",
        "observation/state",
        "actions",
        "prompt",
    )
    for name in required:
        if name not in sample:
            raise ValueError(f"Missing required OpenPI input: {name}")

    state = np.asarray(sample["observation/state"])
    if state.shape != (19,) or not np.issubdtype(state.dtype, np.number):
        raise ValueError(
            f"observation/state must be numeric (19,), "
            f"got {state.shape}, {state.dtype}"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError("observation/state contains nonfinite values")

    action = np.asarray(sample["actions"])
    if (
        action.ndim != 2
        or action.shape[0] < 1
        or action.shape[1] != 7
        or not np.issubdtype(action.dtype, np.number)
    ):
        raise ValueError(
            f"actions must be numeric (N, 7), N >= 1; "
            f"got {action.shape}, {action.dtype}"
        )
    if not np.all(np.isfinite(action)):
        raise ValueError("actions contains nonfinite values")

    for name in ("observation/image", "observation/extra_view_image"):
        frame = np.asarray(sample[name])
        if (
            frame.ndim != 3
            or frame.shape[-1] != 3
            or frame.shape[0] == 0
            or frame.shape[1] == 0
            or frame.dtype != np.uint8
        ):
            raise ValueError(
                f"{name} must be HWC uint8 with 3 channels, "
                f"got {frame.shape}, {frame.dtype}"
            )

    if not isinstance(sample["prompt"], (str, bytes)) or not sample["prompt"]:
        raise ValueError("prompt must be nonempty text")
