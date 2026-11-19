#!/usr/bin/env python
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

"""Restricted external SSH edge fake; it opens no network or ROS connection."""

import json
import os
import sys
import time
from pathlib import Path

Path(os.environ["RLINF_TEST_SSH_PID"]).write_text(str(os.getpid()))
assert len(sys.argv) == 12
assert sys.argv[1:12] == [
    "-T",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "ServerAliveInterval=5",
    "-o",
    "ServerAliveCountMax=2",
    "nuc-franka",
    sys.argv[11],
]
assert "source /opt/ros/humble/setup.bash" in sys.argv[11]
assert "/usr/bin/python3 -s -u -c " in sys.argv[11]
mode = os.environ.get("RLINF_TEST_SSH_MODE", "normal")
if mode == "hang":
    time.sleep(30)
request = json.load(sys.stdin)
assert set(request) == {"schema", "nonce", "modules"}
assert set(request["modules"]) == {"protocol", "passive", "remote_probe"}
assert request["schema"] == "fr3_nuc_probe_request_v1"
if mode == "exit":
    sys.exit(255)
if mode == "overflow":
    sys.stdout.write("x" * (600 * 1024))
    sys.exit(0)
if mode == "malformed":
    sys.stdout.write("invalid JSON")
    sys.exit(0)
result = json.loads(Path(os.environ["RLINF_TEST_NUC_EVIDENCE"]).read_text())
result["nonce"] = "bad" if mode == "nonce" else request["nonce"]
if mode == "schema":
    result["schema"] = "unknown"
if mode == "stderr":
    sys.stderr.write("diagnostic" * 32768)
sys.stdout.write(json.dumps(result, allow_nan=False))
