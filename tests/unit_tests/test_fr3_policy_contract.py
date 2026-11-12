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

"""CPU-only Milestone 4C.2 proposed contract checks."""

import numpy as np
import pytest

from toolkits.realworld_check.fr3_policy_contract import (
    ACTION_FIELDS, CONTRACT_VERSION, FEATURE_NAMES, candidate_19d_state,
)


def state():
    return {
        "tcp_pose": [0.1, 0.2, 0.3, 0, 0, 0, 1],
        "arm_joint_position": [1, 2, 3, 4, 5, 6, 7],
        "arm_joint_velocity": [10] * 7,
        "arm_joint_effort": [20] * 7,
        "tcp_force": [8, 9, 10],
        "tcp_torque": [3, 4, 0],
        "robotiq_knuckle_position": [0.4],
    }


def test_candidate_explicit_order_19d_and_torque_norm():
    result = candidate_19d_state(state())
    assert result.shape == (19,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:7], [0.1, 0.2, 0.3, 0, 0, 0, 1])
    np.testing.assert_allclose(result[7:14], [1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(result[14:17], [8, 9, 10])
    assert result[17] == pytest.approx(0.4)
    assert result[18] == pytest.approx(5.0)
    assert len(FEATURE_NAMES) == 19


def test_candidate_does_not_silently_use_joint_velocity_or_effort():
    original = candidate_19d_state(state())
    other = state()
    other["arm_joint_velocity"] = [100] * 7
    other["arm_joint_effort"] = [200] * 7
    np.testing.assert_array_equal(original, candidate_19d_state(other))


@pytest.mark.parametrize("field,n", [
    ("tcp_pose", 6), ("arm_joint_position", 6),
    ("tcp_force", 4), ("tcp_torque", 2),
    ("robotiq_knuckle_position", 0),
])
def test_bad_dimension_rejected(field, n):
    bad = state()
    bad[field] = [0] * n
    with pytest.raises(ValueError, match=field):
        candidate_19d_state(bad)


def test_nonfinite_gripper_rejected():
    bad = state()
    bad["robotiq_knuckle_position"] = [float("nan")]
    with pytest.raises(ValueError, match="robotiq"):
        candidate_19d_state(bad)


def test_nonunit_quaternion_rejected():
    bad = state()
    bad["tcp_pose"] = [0.1, 0.2, 0.3, 0, 0, 0, 2]
    with pytest.raises(ValueError, match="quaternion"):
        candidate_19d_state(bad)


def test_missing_torque_rejected():
    bad = state()
    del bad["tcp_torque"]
    with pytest.raises(ValueError, match="tcp_torque"):
        candidate_19d_state(bad)


def test_action_contract_is_metadata_only_and_has_7_fields():
    assert CONTRACT_VERSION.endswith("_candidate_v2")
    assert ACTION_FIELDS == (
        "delta_tcp_x_m", "delta_tcp_y_m", "delta_tcp_z_m",
        "delta_tcp_rotvec_x_rad", "delta_tcp_rotvec_y_rad",
        "delta_tcp_rotvec_z_rad", "gripper_target_knuckle_rad",
    )
