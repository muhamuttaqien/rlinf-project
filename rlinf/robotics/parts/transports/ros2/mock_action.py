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

"""A fixed offline subprocess transport using Milestone 2A's bounded local IPC."""

import math
import os
import socket
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .action_contract import InterfaceIdentity
from .protocol import receive_packet, send_packet


class MockROS2ActionTransport:
    """Own stdlib-only fake endpoints; there is no live transport configuration.

    Args:
        timeout: Positive wall-time bound on worker startup and each IPC request.
    """

    def __init__(self, timeout: float = 1.0) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Expected positive finite IPC timeout")
        self.timeout = timeout
        self._lock = threading.Lock()
        self._channel: socket.socket | None = None
        self._process: subprocess.Popen | None = None

    @property
    def is_alive(self) -> bool:
        """Return whether this local mock worker is alive."""
        return self._process is not None and self._process.poll() is None

    def start(
        self, interface: InterfaceIdentity, session_id: str, authority_id: str
    ) -> None:
        """Start disarmed endpoints with no ROS runtime, domain, or network socket."""
        with self._lock:
            self._close()
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self._channel = parent
            parent.settimeout(self.timeout)
            try:
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith(
                        ("ROS_", "RMW_", "FASTRTPS_", "FASTDDS_", "CYCLONEDDS_")
                    )
                    and key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
                }
                self._process = subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-u",
                        str(Path(__file__).with_name("mock_action_worker.py")),
                        "--fd",
                        str(child.fileno()),
                    ],
                    env=env,
                    pass_fds=(child.fileno(),),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                child.close()
                ready = receive_packet(parent)
                if ready != {
                    "version": 1,
                    "op": "ready",
                    "offline": True,
                    "ros_domain_id": None,
                }:
                    raise ConnectionError("Expected offline-only mock readiness")
                send_packet(
                    parent,
                    {
                        "op": "configure",
                        "interface": asdict(interface),
                        "session_id": session_id,
                        "authority_id": authority_id,
                    },
                )
                self._receive("configure")
            except BaseException:
                child.close()
                self._close()
                raise

    def exchange(self, op: str, **payload: Any) -> dict[str, Any]:
        """Exchange one bounded packet; invalidate IPC on any failed exchange."""
        with self._lock:
            try:
                if not self.is_alive or self._channel is None:
                    raise ConnectionError("Expected a running mock action worker")
                send_packet(self._channel, {"op": op, **payload})
                return self._receive(op)
            except (OSError, ValueError, ConnectionError) as exc:
                self._close()
                raise ConnectionError(f"Mock action IPC failed: {exc}") from exc

    def _receive(self, op: str) -> dict[str, Any]:
        response = receive_packet(self._channel)
        if (
            response.get("op") != op
            or response.get("offline") is not True
            or response.get("ros_domain_id") is not None
        ):
            raise ConnectionError("Expected matching offline-only mock response")
        return response

    def close(self) -> None:
        """Close IPC and reap only the owned fake worker, idempotently."""
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self._channel is not None:
            try:
                self._channel.settimeout(0.1)
                if self.is_alive:
                    send_packet(self._channel, {"op": "shutdown"})
                    receive_packet(self._channel)
            except (OSError, ValueError, ConnectionError):
                pass
            finally:
                self._channel.close()
                self._channel = None
        if self._process is not None:
            try:
                self._process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            self._process = None
