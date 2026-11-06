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

"""CPU-only acceptance criteria for the live 3B monitor."""

from collections import Counter

import pytest

from toolkits.realworld_check.fr3_live_qualification import summarize_qualification


def counts(value=58):
    return dict.fromkeys(
        ("joint_state", "tcp_pose", "external_wrench", "gripper_state", "base", "wrist"),
        value,
    )


def test_live_acceptance_succeeds_with_fresh_distinct_sources():
    report = summarize_qualification(
        samples=60, valid=59, distinct=counts(), skews=[0.02] * 59,
        errors=Counter(), elapsed_s=4.0
    )
    assert report["passed"]
    assert report["valid_fraction"] == pytest.approx(59 / 60)
    assert report["p95_camera_receipt_skew_s"] == pytest.approx(0.02)
    assert report["distinct_fraction"]["gripper_state"] == pytest.approx(58 / 60)


def test_live_acceptance_rejects_invalid_or_repeated_feedback():
    report = summarize_qualification(
        samples=60, valid=55, distinct=counts(59), skews=[0.01] * 55,
        errors=Counter({"stale": 5}), elapsed_s=4.0
    )
    assert not report["passed"]
    report = summarize_qualification(
        samples=60, valid=60, distinct={**counts(), "gripper_state": 1},
        skews=[0.01] * 60, errors=Counter(), elapsed_s=4.0
    )
    assert not report["passed"]
    assert not report["checks"]["distinct_sources"]


def test_live_acceptance_rejects_missing_skew_records():
    report = summarize_qualification(
        samples=60, valid=59, distinct=counts(),
        skews=[0.01] * 58, errors=Counter(), elapsed_s=4.0
    )
    assert not report["passed"]


def test_live_acceptance_rejects_invalid_options():
    with pytest.raises(ValueError, match="positive"):
        summarize_qualification(
            samples=0, valid=0, distinct={}, skews=[],
            errors=Counter(), elapsed_s=1
        )
