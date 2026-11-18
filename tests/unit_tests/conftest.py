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

"""Make repository and shared test modules importable by unit tests."""

import sys
from pathlib import Path

import pytest

for _path in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture
def fr3_mapping_verification(tmp_path):
    """Synthetic attestation for fake ROS tests; never a physical approval."""
    import hashlib
    import json

    from toolkits.realworld_check.fr3_pi05_request_adapter import (
        ARM_AXES,
        MAPPING_CHECKS,
    )

    evidence = tmp_path / "OFFLINE_FAKE_ROS_EVIDENCE.txt"
    evidence.write_text(
        "Synthetic test evidence. No physical verification performed.\n"
    )
    record = {
        "schema": "fr3_pi05_mapping_verification_v1",
        "mapping_mode": "identity_verified",
        "operator": "OFFLINE TEST FIXTURE",
        "verified_at_utc": "2020-01-01T00:00:00Z",
        "controller_binary_sha256": "a" * 64,
        "controller_config_sha256": "b" * 64,
        "base_frame": "test_base",
        "tool_frame": "test_tcp",
        "arm_axes": list(ARM_AXES),
        "policy_period_ns": 100_000_000,
        "publish_period_ns": 10_000_000,
        "checks": {
            name: {
                "passed": True,
                "evidence_path": evidence.name,
                "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            }
            for name in MAPPING_CHECKS
        },
    }
    path = tmp_path / "OFFLINE_TEST_MAPPING.json"
    path.write_text(json.dumps(record))
    return path, record
