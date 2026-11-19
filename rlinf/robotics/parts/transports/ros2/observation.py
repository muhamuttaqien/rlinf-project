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

"""Local process ownership for a ROS 2 observation-only subscriber."""

import math
import os
import socket
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Any

from .protocol import PROFILES, receive_packet, send_packet


class ROS2ObservationTransport:
    """Own one ROS-compatible worker and a serialized, bounded IPC connection.

    Args:
        profile: Fixed state subscription set: ``fr3`` or ``robotiq``.
        domain_id: Explicit DDS domain for this worker.
        ros_python: Absolute path to the ROS-compatible Python executable.
        ros_setup: Absolute ROS setup.bash path, sourced in an isolated shell.
        timeout: Positive startup and per-request timeout in seconds.
    """

    def __init__(
        self,
        profile: str,
        domain_id: int,
        *,
        ros_python: str = "/usr/bin/python3",
        ros_setup: str = "/opt/ros/jazzy/setup.bash",
        timeout: float = 5.0,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(f"Expected profile in {tuple(PROFILES)}; got {profile!r}")
        if type(domain_id) is not int or not 0 <= domain_id <= 232:
            raise ValueError(f"Expected ROS domain ID in [0, 232]; got {domain_id!r}")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"Expected positive finite timeout; got {timeout}")
        for value in (ros_python, ros_setup):
            if not Path(value).is_absolute():
                raise ValueError(f"Expected an absolute runtime path; got {value!r}")
        self.profile, self.domain_id = profile, domain_id
        self.ros_python, self.ros_setup, self.timeout = ros_python, ros_setup, timeout
        self._lock = threading.Lock()
        self._channel: socket.socket | None = None
        self._process: subprocess.Popen | None = None
        self._stderr: deque[bytes] = deque(maxlen=8)
        self._drainer: threading.Thread | None = None
        self._logs: tempfile.TemporaryDirectory | None = None

    @property
    def is_alive(self) -> bool:
        """Whether the owned worker process is running."""
        return self._process is not None and self._process.poll() is None

    @property
    def diagnostics(self) -> str:
        """Return the last 16 KiB of worker stderr, including runtime errors."""
        return b"".join(tuple(self._stderr)).decode(errors="replace")

    def start(self) -> None:
        """Start the subscriber; roll back all resources on startup failure."""
        with self._lock:
            if self.is_alive:
                return
            self._close()
            self._stderr.clear()
            for value in (self.ros_python, self.ros_setup):
                if not Path(value).is_file():
                    raise ConnectionError(
                        f"Expected an existing ROS runtime file; got {value!r}"
                    )
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self._channel = parent
            parent.settimeout(self.timeout)
            env = dict(os.environ)
            for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
                env.pop(key, None)
            env["ROS_DOMAIN_ID"] = str(self.domain_id)
            worker = Path(__file__).with_name("worker.py")
            command = [
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                'source "$1" && exec "$2" -s -u "$3" --fd "$4" --domain-id "$5" --profile "$6"',
                "rlinf-ros2-observations",
                self.ros_setup,
                self.ros_python,
                str(worker),
                str(child.fileno()),
                str(self.domain_id),
                self.profile,
            ]
            try:
                self._logs = tempfile.TemporaryDirectory(prefix="rlinf-ros2-")
                env["ROS_LOG_DIR"] = self._logs.name
                self._process = subprocess.Popen(
                    command,
                    env=env,
                    pass_fds=(child.fileno(),),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                child.close()
                self._drainer = threading.Thread(target=self._drain, daemon=True)
                self._drainer.start()
                ready = receive_packet(parent)
                if (ready.get("op"), ready.get("domain_id"), ready.get("profile")) != (
                    "ready",
                    self.domain_id,
                    self.profile,
                ):
                    raise ConnectionError(f"Expected worker readiness; got {ready}")
            except BaseException as exc:
                child.close()
                self._close()
                if isinstance(exc, Exception):
                    raise ConnectionError(
                        f"ROS observation worker startup failed: {exc}. {self.diagnostics}"
                    ) from exc
                raise

    def _drain(self) -> None:
        process = self._process
        if process is not None and process.stderr is not None:
            while chunk := os.read(process.stderr.fileno(), 2048):
                self._stderr.append(chunk)

    def snapshot(self) -> dict[str, Any]:
        """Request the latest samples; close the session on any IPC failure."""
        return self._request("snapshot")

    def inspect_status(self, budget: float = 2.0) -> dict[str, Any]:
        """Collect fixed read-only graph/status queries while subscriptions spin.

        Budget is a shared ROS discovery/response window, not a latency bound.
        The IPC timeout must exceed it by at least 0.5 seconds.
        """
        if not math.isfinite(budget) or not 0 < budget <= min(5.0, self.timeout - 0.5):
            raise ValueError("Expected diagnostic budget in (0, min(5, timeout-0.5)]")
        return self._request("inspect_status", budget=budget)

    def _request(self, operation: str, **settings: Any) -> dict[str, Any]:
        with self._lock:
            try:
                if not self.is_alive or self._channel is None:
                    raise ConnectionError("Expected a running ROS observation worker")
                send_packet(self._channel, {"op": operation, **settings})
                result = receive_packet(self._channel)
                if (
                    result.get("op"),
                    result.get("domain_id"),
                    result.get("profile"),
                ) != (operation, self.domain_id, self.profile):
                    raise ConnectionError(
                        f"Expected domain-specific snapshot; got {result}"
                    )
                return result
            except (OSError, ValueError, ConnectionError) as exc:
                self._close()
                raise ConnectionError(
                    f"ROS observation IPC failed: {exc}. {self.diagnostics}"
                ) from exc

    def close(self) -> None:
        """Request graceful exit, then terminate/reap a nonresponsive worker."""
        with self._lock:
            self._close()

    def _close(self) -> None:
        channel, process = self._channel, self._process
        if channel is not None:
            try:
                channel.settimeout(0.5)
                if self.is_alive:
                    send_packet(channel, {"op": "shutdown"})
                    receive_packet(channel)
            except (OSError, ValueError, ConnectionError):
                pass
            finally:
                channel.close()
                self._channel = None
        if process is not None:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if self._drainer is not None:
                self._drainer.join(timeout=1.0)
            if process.stderr is not None:
                process.stderr.close()
        self._process = self._drainer = None
        if self._logs is not None:
            self._logs.cleanup()
            self._logs = None
