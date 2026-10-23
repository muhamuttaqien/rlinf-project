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

import numpy as np
import pytest

from toolkits.realworld_check.franka_pi05_contract import validate_realworld_pi05_sample


def sample():
    return {
        "observation/image": np.zeros((128, 128, 3), dtype=np.uint8),
        "observation/extra_view_image": np.zeros((128, 128, 3), dtype=np.uint8),
        "observation/state": np.zeros(19, dtype=np.float32),
        "actions": np.zeros((5, 7), dtype=np.float32),
        "prompt": "pick up the object",
    }


def test_valid_transform_sample():
    validate_realworld_pi05_sample(sample())


def test_bad_state_length_rejected():
    s = sample()
    s["observation/state"] = np.zeros(20, dtype=np.float32)
    with pytest.raises(ValueError, match="observation/state"):
        validate_realworld_pi05_sample(s)


def test_bad_action_width_rejected():
    s = sample()
    s["actions"] = np.zeros((5, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="actions"):
        validate_realworld_pi05_sample(s)


def test_nonfinite_actions_rejected():
    s = sample()
    s["actions"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_realworld_pi05_sample(s)


def test_bad_image_rejected():
    s = sample()
    s["observation/image"] = np.zeros((128, 128), dtype=np.uint8)
    with pytest.raises(ValueError, match="observation/image"):
        validate_realworld_pi05_sample(s)


def test_missing_extra_view_accepted_as_upstream_fallback():
    s = sample()
    del s["observation/extra_view_image"]
    validate_realworld_pi05_sample(s)
