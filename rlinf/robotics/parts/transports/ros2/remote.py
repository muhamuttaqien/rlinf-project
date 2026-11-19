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

"""Bounded SSH transport for the fixed, installation-free NUC passive probe."""

import json
import math
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

MAX_OUTPUT = 512 * 1024
MAX_INPUT = 192 * 1024
BOOTSTRAP = """import json,re,sys,types
packet=json.loads(sys.stdin.buffer.read(196609))
if packet.get("schema") != "fr3_nuc_probe_request_v1" or not re.fullmatch("[0-9a-f]{32}", packet.get("nonce", "")):
    raise ValueError("Expected fixed passive probe request and nonce")
if set(packet.get("modules", {})) != {"protocol", "passive", "remote_probe"}:
    raise ValueError("Expected only fixed passive probe modules")
for name in ("protocol", "passive", "remote_probe"):
    module=types.ModuleType(name)
    sys.modules[name]=module
    exec(compile(packet["modules"][name], "<rlinf_"+name+">", "exec"), module.__dict__)
result=sys.modules["remote_probe"].collect(packet["nonce"])
encoded=json.dumps(result,allow_nan=False)
if len(encoded.encode()) > 524288:
    raise ValueError("Expected passive evidence <=512 KiB")
sys.stdout.write(encoded+"\\n")
"""
# Host, runtime and setup paths are reviewed constants, never CLI/RPC input.
REMOTE_COMMAND = (
    "source /opt/ros/humble/setup.bash && "
    "source /home/aist-franka/Desktop/aist-gamma/franka_ros2_ws/install/franka_msgs/share/franka_msgs/local_setup.bash && "
    "source /home/aist-franka/Desktop/aist-gamma/robotiq_ros2_ws/install/robotiq_msgs/share/robotiq_msgs/local_setup.bash && "
    "/usr/bin/python3 -s -u -c "
)


def collect_nuc_evidence(timeout: float = 35.0) -> dict[str, Any]:
    """Run only the fixed passive program on authorized alias nuc-franka.

    Each invocation transfers repository-owned modules through stdin. No remote
    files, agents, arbitrary command/service/host arguments or actions are used.
    A nonce binds the bounded response to this invocation, not to physical trust.
    """
    import shlex

    if not math.isfinite(timeout) or not 0.05 <= timeout <= 35:
        raise ValueError("Expected passive SSH timeout in [0.05,35] seconds")
    nonce = uuid.uuid4().hex
    directory = Path(__file__).parent
    request = json.dumps(
        {
            "schema": "fr3_nuc_probe_request_v1",
            "nonce": nonce,
            "modules": {
                name: directory.joinpath(name + ".py").read_text()
                for name in ("protocol", "passive", "remote_probe")
            },
        }
    ).encode()
    if len(request) > MAX_INPUT:
        raise ValueError("Expected passive probe bundle <=192 KiB")
    command = [
        "ssh",
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
        REMOTE_COMMAND + shlex.quote(BOOTSTRAP),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, errors = bytearray(), bytearray()
    overflow = threading.Event()

    def drain(stream: Any, target: bytearray, limit: int) -> None:
        while chunk := stream.read(4096):
            room = limit - len(target)
            target.extend(chunk[: max(0, room)])
            if len(chunk) > room and target is output:
                overflow.set()
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

    threads = [
        threading.Thread(target=drain, args=(stream, target, limit), daemon=True)
        for stream, target, limit in (
            (process.stdout, output, MAX_OUTPUT),
            (process.stderr, errors, 16384),
        )
    ]

    def write_request() -> None:
        try:
            process.stdin.write(request)
        except OSError:
            pass
        finally:
            process.stdin.close()

    threads.append(threading.Thread(target=write_request, daemon=True))
    try:
        for thread in threads:
            thread.start()
        process.wait(timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        process.kill()
        raise ConnectionError(f"NUC passive SSH timeout/IPC failure: {exc}") from exc
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
    if overflow.is_set():
        raise ConnectionError("NUC passive response exceeds 512 KiB")
    if process.returncode != 0:
        raise ConnectionError(
            f"NUC passive SSH exited {process.returncode}: {errors.decode(errors='replace')[:2048]}"
        )
    try:
        result = json.loads(output)
        if (
            not isinstance(result, dict)
            or result.get("schema") != "fr3_nuc_passive_v1"
            or result.get("nonce") != nonce
        ):
            raise ValueError(
                "Expected matching versioned NUC evidence and invocation nonce"
            )
    except (ValueError, UnicodeError) as exc:
        raise ConnectionError(f"Invalid NUC passive JSON evidence: {exc}") from exc
    return result
