# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prepare or execute one explicitly confirmed, bounded FR3 calibration case."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import select
import signal
import sys
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Protocol

from toolkits.realworld_check.fr3_pi05_request_adapter import (
    ARM_AXES,
    MAPPING_CHECKS,
    GripperRequestFilter,
)

NAME = "rlinf_fr3_mapping_calibration"
ARM_TOPIC = "/franka_controller/target_cartesian_velocity_percent"
GRIPPER_TOPIC = "/gripper_client/target_gripper_width_percent"
PERIOD_NS = 100_000_000
REFRESH_NS = 10_000_000
MAX_GAP_NS = 50_000_000
POSE_AGE_NS = 200_000_000
BASELINE_NS = 2_100_000_000
POST_NS = 2_100_000_000
AXIS_CASES = MAPPING_CHECKS[:12]
CASES = (*AXIS_CASES, "gripper_open", "gripper_close", "gripper_hold", "neutral")
DIAG_FIELDS = (
    "rx",
    "moving_rx",
    "invalid_rx",
    "ik_attempt",
    "ik_valid",
    "ik_failed",
    "gate_ticks",
    "ref_advanced_ticks",
    "effort_write_ticks",
    "peak_joint_error_rad",
    "peak_pd_effort_nm",
    "peak_command_effort_nm",
)


class Blocked(RuntimeError):
    """A calibration prerequisite or live guard failed."""


class NotReady(Blocked):
    """Read-only discovery has not completed."""


class OperatorStop(Blocked):
    """The operator revoked the current case's command authority."""


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: dict) -> str:
    """Bind consent to a canonical plan and installation scope."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def finite(values: list | tuple, length: int) -> tuple[float, ...]:
    """Require a finite numeric vector of the expected length."""
    if len(values) != length or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in values
    ):
        raise Blocked(f"Expected {length} finite numeric values, found {values!r}")
    return tuple(float(x) for x in values)


def quaternion(values: list | tuple) -> tuple[float, ...]:
    """Validate an xyzw quaternion and remove small normalization error."""
    q = finite(values, 4)
    norm = math.sqrt(sum(x * x for x in q))
    if abs(norm - 1) > 0.001:
        raise Blocked(f"Expected unit quaternion, found norm {norm}")
    return tuple(x / norm for x in q)


def multiply(a: tuple, b: tuple) -> tuple:
    """Multiply xyzw quaternions."""
    x, y, z, w = a
    u, v, t, s = b
    return (
        w * u + x * s + y * t - z * v,
        w * v - x * t + y * s + z * u,
        w * t + x * v - y * u + z * s,
        w * s - x * u - y * v - z * t,
    )


def rotate(q: tuple, p: tuple) -> tuple:
    """Rotate a vector by a unit xyzw quaternion."""
    return multiply(multiply(q, (*p, 0)), (-q[0], -q[1], -q[2], q[3]))[:3]


def pose(values: list | tuple) -> tuple:
    """Validate a position plus xyzw orientation."""
    p = finite(values, 7)
    return (*p[:3], *quaternion(p[3:]))


def compose(a: tuple, b: tuple) -> tuple:
    """Compose parent-to-child rigid poses."""
    offset = rotate(a[3:], b[:3])
    return (*[a[i] + offset[i] for i in range(3)], *multiply(a[3:], b[3:]))


def rotation_vector(a: tuple, b: tuple) -> tuple:
    """Return the shortest rotation from a to b in a's local axes."""
    q = multiply((-a[3], -a[4], -a[5], a[6]), b[3:])
    if q[3] < 0:
        q = tuple(-x for x in q)
    size = math.sqrt(sum(x * x for x in q[:3]))
    factor = 2 * math.atan2(size, q[3]) / size if size else 2.0
    return tuple(x * factor for x in q[:3])


def message_pose(message: dict) -> tuple:
    """Read a geometry_msgs/PoseStamped dictionary."""
    p, q = message["pose"]["position"], message["pose"]["orientation"]
    return pose([p[k] for k in ("x", "y", "z")] + [q[k] for k in ("x", "y", "z", "w")])


@dataclass(frozen=True)
class Plan:
    """One case; five distinguishable 100 ms arm slots or one gripper goal."""

    case: str
    request: float = 0.03
    duration_ns: int = 500_000_000

    def __post_init__(self) -> None:
        if self.case not in CASES:
            raise Blocked(f"Expected one supported case, found {self.case!r}")
        if (
            type(self.request) not in (float, int)
            or not math.isfinite(self.request)
            or not 0 < self.request <= 0.05
        ):
            raise Blocked("Expected normalized request magnitude in (0, 0.05]")
        if type(self.duration_ns) is not int or self.duration_ns not in range(
            PERIOD_NS, 500_000_001, PERIOD_NS
        ):
            raise Blocked("Expected duration 0.1..0.5 s in whole 100 ms slots")
        if (
            self.case in ("gripper_open", "gripper_close")
            and self.duration_ns < 3 * PERIOD_NS
        ):
            raise Blocked(
                "Open/close calibration requires at least three policy samples"
            )

    @property
    def gripper(self) -> bool:
        """Whether the case exercises the gripper request filter."""
        return self.case.startswith("gripper_")

    def sample(self, index: int) -> tuple[list[float], float | None]:
        """Return one normalized request, without physical-unit conversion."""
        arm = [0.0] * 6
        if self.case in AXIS_CASES:
            axis, direction = self.case.rsplit("_", 1)
            # Distinct slots expose a stale-held-sample error in the sender trace.
            arm[ARM_AXES.index(axis)] = (
                self.request
                * (1, 0.8, 0.6, 0.8, 1)[index]
                * (1 if direction == "positive" else -1)
            )
        grip = {"gripper_open": 1.0, "gripper_close": 0.0, "gripper_hold": 0.5}.get(
            self.case
        )
        return arm, grip


def load_deployment(path: Path) -> dict:
    """Verify copied installation artifacts and explicit operator preparation.

    Local hashes cannot identify the live NUC process. The operator must match
    that installation again at the case-specific console confirmation.
    """
    raw = json.loads(path.read_text())
    if raw.get("schema") != "fr3_mapping_calibration_deployment_v1":
        raise Blocked("Expected calibration deployment schema v1")
    for name in (
        "operator",
        "base_frame",
        "pose_frame",
        "tool_frame",
        "diagnostic_logger",
        "gripper_joint",
        "gripper_bridge_node",
    ):
        if not isinstance(raw.get(name), str) or not raw[name].strip():
            raise Blocked(f"Expected explicit deployment field: {name}")
    for name in (
        "live_installation_matched",
        "frames_calibrated",
        "hardware_stop_checked",
        "workspace_clear",
        "historical_errors_reviewed",
    ):
        if raw.get(name) is not True:
            raise Blocked(f"Operator preparation not confirmed: {name}")
    for name, limit in (("translation_limit_m", 0.01), ("rotation_limit_rad", 0.01)):
        value = raw.get(name)
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or not 0 < value <= limit
        ):
            raise Blocked(f"Expected {name} in (0, {limit}], found {value!r}")
    for name in ("base_to_pose", "ee_to_flange", "ee_to_tcp"):
        pose(raw[name])
    files = []
    for name in ("controller_binary", "controller_config"):
        item = raw[name]
        source = (path.parent / item["path"]).resolve()
        if (
            not source.is_file()
            or source.stat().st_size == 0
            or sha256(source) != item["sha256"]
        ):
            raise Blocked(f"Copied deployment artifact hash mismatch: {name}")
        files.append((source, item["sha256"]))
    config = json.loads(files[1][0].read_text())
    if config.get("schema") != "fr3_mapping_config_manifest_v1" or not config.get(
        "files"
    ):
        raise Blocked(
            "Expected nonempty config manifest with captured runtime parameters, URDF and tool evidence"
        )
    for item in config["files"]:
        source = (files[1][0].parent / item["path"]).resolve()
        if (
            not source.is_file()
            or source.stat().st_size == 0
            or sha256(source) != item["sha256"]
        ):
            raise Blocked(f"Configuration evidence hash mismatch: {source}")
        files.append((source, item["sha256"]))
    scope = deployment_scope(raw)
    return {
        "record": raw,
        "files": files,
        "sha256": sha256(path),
        "scope": scope,
        "scope_sha256": json_digest(scope),
    }


def deployment_scope(raw: dict) -> dict:
    """Identify the controller/configuration and calibrated physical frame chain."""
    scope = {
        k: raw[k]
        for k in (
            "base_frame",
            "pose_frame",
            "tool_frame",
            "base_to_pose",
            "ee_to_flange",
            "ee_to_tcp",
            "gripper_joint",
        )
    }
    scope.update(
        {
            k + "_sha256": raw[k]["sha256"]
            for k in ("controller_binary", "controller_config")
        }
    )
    return scope


def check_endpoints(
    publishers: list, subscribers: list, *, own: bool, topic_type: str, peer: str | None
) -> None:
    """Require one known subscriber and exclusive calibration ownership."""
    if publishers and (
        not own or len(publishers) != 1 or publishers[0].node_name != NAME
    ):
        raise Blocked("Competing command publisher detected")
    if own and not publishers:
        raise NotReady("Calibration publisher not discovered")
    if not subscribers:
        raise NotReady("Expected one command subscriber, found zero")
    if len(subscribers) != 1:
        raise Blocked(
            f"Expected exactly one command subscriber, found {len(subscribers)}"
        )
    endpoint = subscribers[0]
    if endpoint.topic_type != topic_type or (
        peer and endpoint.node_name not in (peer, "_NODE_NAME_UNKNOWN_")
    ):
        raise Blocked("Unexpected command subscriber type or identity")


class Telemetry:
    """Keep source receipts and validate frames, state, faults and freshness."""

    def __init__(self) -> None:
        self.latest: dict[str, dict] = {}
        self.diag_count = 0

    def receive(self, source: str, message: dict, now_ns: int) -> None:
        """Store one complete message with its host receipt time."""
        if source == "diagnostics" and source in self.latest:
            old = self.latest[source]["message"]
            if any(message[k] < old[k] for k in DIAG_FIELDS[:9]):
                raise Blocked("Controller diagnostic counters reset during calibration")
        self.latest[source] = {"message": message, "received_monotonic_ns": now_ns}
        if source == "diagnostics":
            self.diag_count += 1

    def check(
        self,
        deployment: dict,
        now_ns: int,
        *,
        gripper: bool,
        origin: dict | None = None,
    ) -> dict:
        """Require fresh state and bound EE, flange and peg TCP motion."""
        required = {
            "pose": POSE_AGE_NS,
            "measured_joints": POSE_AGE_NS,
            "desired_joints": POSE_AGE_NS,
            "robot_state": POSE_AGE_NS,
            "controller": 1_000_000_000,
            "diagnostics": 2_000_000_000,
            "gripper_state": POSE_AGE_NS,
        }
        # Action status is an event stream, not a periodic state heartbeat.
        for name, age in required.items():
            item = self.latest.get(name)
            if item is None or not 0 <= now_ns - item["received_monotonic_ns"] <= age:
                raise NotReady(f"Missing or stale telemetry: {name}")
        raw = deployment["record"]
        pose_message = self.latest["pose"]["message"]
        if pose_message["header"]["frame_id"] != raw["pose_frame"]:
            raise Blocked("Measured pose frame differs from the approved frame")
        ee = compose(pose(raw["base_to_pose"]), message_pose(pose_message))
        result = {
            "ee": ee,
            "flange": compose(ee, pose(raw["ee_to_flange"])),
            "tcp": compose(ee, pose(raw["ee_to_tcp"])),
        }
        for name in ("measured_joints", "desired_joints"):
            m = self.latest[name]["message"]
            if m["name"] != [f"fr3_joint{i}" for i in range(1, 8)]:
                raise Blocked(f"Expected seven ordered FR3 joints in {name}")
            for field in ("position", "velocity", "effort"):
                finite(m[field], 7)
        robot = self.latest["robot_state"]["message"]
        if (
            robot["robot_mode"] not in (1, 2)
            or not robot["current_errors"]
            or any(type(x) is not bool or x for x in robot["current_errors"].values())
        ):
            raise Blocked("FR3 mode or current_errors does not permit calibration")
        finite([robot["control_command_success_rate"]], 1)
        # The installed robot-state transform must agree with the approved tool chain.
        identity = compose(message_pose(robot["f_t_ee"]), pose(raw["ee_to_flange"]))
        if (
            math.dist(identity[:3], (0, 0, 0)) > 1e-6
            or math.dist(rotation_vector((0, 0, 0, 0, 0, 0, 1), identity), (0, 0, 0))
            > 1e-6
        ):
            raise Blocked("Robot flange-to-EE transform differs from approved geometry")
        controller = self.latest["controller"]["message"]
        if controller["state"] != "active" or set(controller["claimed_interfaces"]) != {
            f"fr3_joint{i}/effort" for i in range(1, 8)
        }:
            raise Blocked(
                "Expected active joint impedance controller claiming seven effort interfaces"
            )
        diag = self.latest["diagnostics"]["message"]
        finite([diag[k] for k in DIAG_FIELDS], len(DIAG_FIELDS))
        grip = self.latest["gripper_state"]["message"]
        if grip["name"].count(raw["gripper_joint"]) != 1:
            raise Blocked("Expected one measured Robotiq joint")
        finite([grip["position"][grip["name"].index(raw["gripper_joint"])]], 1)
        if origin:
            for frame, value in result.items():
                if math.dist(origin[frame][:3], value[:3]) > raw["translation_limit_m"]:
                    raise Blocked(
                        f"Measured {frame} translation exceeded calibration guard"
                    )
                if (
                    math.dist(rotation_vector(origin[frame], value), (0, 0, 0))
                    > raw["rotation_limit_rad"]
                ):
                    raise Blocked(
                        f"Measured {frame} rotation exceeded calibration guard"
                    )
        return result


class Journal:
    """Own a bounded asynchronous evidence writer; overflow revokes motion."""

    def __init__(self, directory: Path, capacity: int = 8192) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.pending: queue.Queue = queue.Queue(maxsize=capacity)
        self.error: Exception | None = None
        self.streams = {
            name: (directory / (name + ".jsonl")).open("x")
            for name in ("events", "telemetry")
        }
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self) -> None:
        try:
            while True:
                item = self.pending.get()
                if item is None:
                    break
                stream, record = item
                self.streams[stream].write(json.dumps(record, allow_nan=False) + "\n")
        except Exception as exc:
            self.error = exc
        finally:
            for stream in self.streams.values():
                try:
                    stream.close()
                except Exception as exc:
                    self.error = exc

    def check(self) -> None:
        """Reject further commands after evidence storage fails."""
        if self.error is not None:
            raise Blocked(f"Evidence writer failed: {self.error}")

    def add(self, stream: str, record: dict) -> None:
        """Queue an immutable event without waiting for storage."""
        self.check()
        try:
            self.pending.put_nowait((stream, record))
        except queue.Full as exc:
            raise Blocked("Evidence queue overflow; calibration aborted") from exc

    def close(self) -> None:
        """Finish owned evidence streams or report an incomplete capture."""
        try:
            self.pending.put(None, timeout=1)
        except queue.Full as exc:
            raise Blocked("Evidence queue could not finish") from exc
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise Blocked("Evidence writer did not finish; capture incomplete")
        self.check()


class Console:
    """Require a physical-session terminal and monitor ENTER/EOF or signals."""

    def __init__(self) -> None:
        self.interrupted = False

    def confirm(self, token: str) -> None:
        """Accept the exact case/installation token only from an interactive TTY."""
        if not sys.stdin.isatty():
            raise Blocked(
                "Interactive operator TTY required; piped approval is forbidden"
            )
        if (
            input(
                f"Match live installation, clear workspace, keep hardware stop ready. Type {token}: "
            )
            != token
        ):
            raise Blocked("Operator did not approve this exact calibration case")

    def stopped(self) -> bool:
        """Revoke authority on ENTER, EOF or a handled signal."""
        if self.interrupted:
            return True
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            self.interrupted = True
        return self.interrupted

    def active(self, plan: Plan) -> None:
        """Tell the operator when the bounded request window is about to start."""
        sys.stderr.write(
            f"Starting {plan.case} for {plan.duration_ns / 1e9:g} s; ENTER stops new commands. Keep hardware stop ready.\n"
        )
        sys.stderr.flush()


def publication_guard(
    now_ns: int, deadline_ns: int, last_ns: int | None, stop: Callable[[], bool]
) -> None:
    """Check consent lifetime immediately before a non-neutral publication."""
    if stop():
        raise OperatorStop("Operator stopped calibration")
    if now_ns >= deadline_ns or (last_ns is not None and now_ns - last_ns > MAX_GAP_NS):
        raise Blocked("Calibration sample expired or 50 ms publication deadline missed")


class CalibrationRuntime(Protocol):
    """ROS/hardware boundary consumed by the bounded calibration loop."""

    backend: str
    clock_id: str
    gripper: bool
    telemetry: Telemetry
    journal: Journal | None
    origin: dict | None

    def now_ns(self) -> int: ...
    def spin(self) -> None: ...
    def check_graph(self, *, own: bool) -> None: ...
    def enable_publishers(self) -> None: ...
    def publish_arm(
        self,
        values: list,
        *,
        deadline_ns: int | None,
        last_ns: int | None,
        stop: Callable[[], bool],
    ) -> int: ...
    def publish_gripper(
        self, value: float, *, deadline_ns: int, last_ns: int, stop: Callable[[], bool]
    ) -> int: ...


@contextmanager
def _local_claim(
    path: Path = Path("/tmp/rlinf_fr3_mapping_calibration.lock"),
) -> Iterator[None]:
    descriptor = os.open(
        path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Blocked(
                "Another local calibration owns the command interface"
            ) from exc
        yield
    finally:
        os.close(descriptor)


def collect_baseline(
    runtime: CalibrationRuntime, deployment: dict, *, timeout_ns: int = 10_000_000_000
) -> None:
    """Wait read-only for continuous fresh telemetry and two diagnostics."""
    deadline = runtime.now_ns() + timeout_ns
    since = None
    diag_start = runtime.telemetry.diag_count
    while runtime.now_ns() < deadline:
        runtime.spin()
        try:
            runtime.check_graph(own=False)
            runtime.telemetry.check(
                deployment, runtime.now_ns(), gripper=runtime.gripper
            )
        except NotReady:
            since = None
        else:
            since = runtime.now_ns() if since is None else since
            if (
                runtime.now_ns() - since >= BASELINE_NS
                and runtime.telemetry.diag_count - diag_start >= 2
            ):
                return
    raise Blocked("Continuous fresh baseline and two diagnostic reports unavailable")


def execute_case(
    runtime: CalibrationRuntime,
    plan: Plan,
    deployment_path: Path,
    directory: Path,
    console: Console,
) -> dict:
    """Run one confirmed case; faults revoke motion and preserve partial evidence.

    The runtime owns ROS resources. A fake runtime can exercise the identical
    state machine offline, but its evidence is explicitly ineligible for review.
    """
    if directory.exists():
        raise Blocked("Calibration evidence directory must be new")
    deployment = load_deployment(deployment_path)
    collect_baseline(runtime, deployment)
    token = f"CALIBRATE {plan.case} {json_digest({'plan': asdict(plan), 'deployment': deployment['sha256'], 'nonce': os.urandom(16).hex()})[:12]}"
    console.confirm(token)
    if load_deployment(deployment_path)["sha256"] != deployment["sha256"]:
        raise Blocked("Deployment changed during operator confirmation")
    with _local_claim() if runtime.backend == "ros" else nullcontext():
        return _capture_case(runtime, plan, deployment, directory, console, token)


def _capture_case(
    runtime: CalibrationRuntime,
    plan: Plan,
    deployment: dict,
    directory: Path,
    console: Console,
    token: str,
) -> dict:
    from toolkits.realworld_check.fr3_mapping_evidence import seal_session, write_json

    journal = Journal(directory)
    runtime.journal = journal
    started = runtime.now_ns()
    summary = {
        "status": "aborted",
        "case": plan.case,
        "backend": runtime.backend,
        "physical_checks_passed": [],
        "commands": 0,
        "gripper_goals": 0,
        "cleanup_neutral_attempts": 0,
        "cleanup_neutral_publications": 0,
    }
    enabled = False
    try:
        inputs = directory / "inputs"
        inputs.mkdir()
        for source, expected in deployment["files"]:
            target = inputs / expected
            if not target.exists():
                target.write_bytes(source.read_bytes())
            if sha256(target) != expected:
                raise Blocked("Deployment artifact changed while snapshotting")
        write_json(
            directory / "session.json",
            {
                "schema": "fr3_mapping_calibration_session_v1",
                "backend": runtime.backend,
                "plan": asdict(plan),
                "deployment": deployment["record"],
                "scope": deployment["scope"],
                "scope_sha256": deployment["scope_sha256"],
                "consent_token": token,
                "clock": "host monotonic_ns; ROS headers retained separately",
                "clock_id": runtime.clock_id,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "started_monotonic_ns": started,
            },
        )
        # Confirmation blocks ROS callbacks; acquire a new continuous baseline.
        collect_baseline(runtime, deployment)
        if console.stopped():
            raise OperatorStop("Stopped before enabling publishers")
        origin = runtime.telemetry.check(
            deployment, runtime.now_ns(), gripper=plan.gripper
        )
        runtime.origin = origin
        runtime.enable_publishers()
        enabled = True
        discovery_end = runtime.now_ns() + 2_000_000_000
        while True:
            runtime.spin()
            journal.check()
            runtime.telemetry.check(
                deployment, runtime.now_ns(), gripper=plan.gripper, origin=origin
            )
            if console.stopped():
                raise OperatorStop("Stopped during publisher discovery")
            try:
                runtime.check_graph(own=True)
                break
            except NotReady:
                if runtime.now_ns() >= discovery_end:
                    raise
        console.active(plan)
        begin = runtime.now_ns()
        last_publish = begin
        next_publish = begin
        last_sample = None
        filter_ = GripperRequestFilter(confirmations=3, min_goal_interval_s=0.8)
        stop_at = None
        journal.add(
            "events", {"event": "case_started", "monotonic_ns": begin, "origin": origin}
        )
        while runtime.now_ns() - begin < plan.duration_ns + POST_NS:
            runtime.spin()
            journal.check()
            runtime.check_graph(own=True)
            runtime.telemetry.check(
                deployment, runtime.now_ns(), gripper=plan.gripper, origin=origin
            )
            now = runtime.now_ns()
            if console.stopped() and stop_at is None:
                stop_at = now
                journal.add("events", {"event": "operator_stop", "monotonic_ns": now})
            if last_publish is not None and now - last_publish > MAX_GAP_NS:
                raise Blocked("Calibration missed its 50 ms transport deadline")
            if now < next_publish:
                continue
            next_publish = begin + ((now - begin) // REFRESH_NS + 1) * REFRESH_NS
            active = stop_at is None and now - begin < plan.duration_ns
            index = (now - begin) // PERIOD_NS if active else None
            arm, grip = plan.sample(index) if active else ([0.0] * 6, None)
            deadline = begin + (index + 1) * PERIOD_NS if active else None
            published = runtime.publish_arm(
                arm, deadline_ns=deadline, last_ns=last_publish, stop=console.stopped
            )
            journal.add(
                "events",
                {
                    "event": "arm_publish",
                    "monotonic_ns": published,
                    "sample_index": index,
                    "deadline_ns": deadline,
                    "request_6d": arm,
                },
            )
            last_publish = published
            if active and index != last_sample:
                last_sample = index
                summary["commands"] += 1
                journal.add(
                    "events",
                    {
                        "event": "sample",
                        "monotonic_ns": published,
                        "index": index,
                        "gripper_request": grip,
                    },
                )
                if grip is not None:
                    goal = filter_.step(grip, now=published / 1e9)
                    if goal != "hold":
                        publication_guard(
                            runtime.now_ns(), deadline, last_publish, console.stopped
                        )
                        goal_at = runtime.publish_gripper(
                            0.0 if goal == "close" else 1.0,
                            deadline_ns=deadline,
                            last_ns=last_publish,
                            stop=console.stopped,
                        )
                        summary["gripper_goals"] += 1
                        journal.add(
                            "events",
                            {
                                "event": "gripper_goal",
                                "monotonic_ns": goal_at,
                                "goal": goal,
                            },
                        )
        summary["status"] = "operator_stop" if stop_at is not None else "completed"
    except (Exception, KeyboardInterrupt) as exc:
        summary["reason"] = str(exc) or type(exc).__name__
    finally:
        # Neutral cleanup is part of the same consent and remains independent of storage.
        if enabled:
            for _ in range(5):
                summary["cleanup_neutral_attempts"] += 1
                try:
                    runtime.publish_arm(
                        [0.0] * 6, deadline_ns=None, last_ns=None, stop=lambda: False
                    )
                    summary["cleanup_neutral_publications"] += 1
                except Exception as exc:
                    summary["status"] = "aborted"
                    summary["cleanup_error"] = str(exc)
        runtime.journal = None
        try:
            journal.close()
        except Exception as exc:
            summary["status"] = "aborted"
            summary["capture_error"] = str(exc)
        summary["ended_monotonic_ns"] = runtime.now_ns()
        summary["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "summary.json", summary)
        seal_session(directory)
    return summary


def main() -> None:
    """Print a plan by default; explicitly select read-only or confirmed ROS use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="linear_x_positive")
    parser.add_argument("--request", type=float, default=0.03)
    parser.add_argument("--duration", type=float, default=0.5)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = Plan(args.case, args.request, round(args.duration * 1e9))
    if not args.preflight and not args.execute:
        sys.stdout.write(
            json.dumps(
                {
                    "plan": asdict(plan),
                    "slots": [
                        plan.sample(i) for i in range(plan.duration_ns // PERIOD_NS)
                    ],
                    "physical_execution_approved": False,
                },
                indent=2,
            )
            + "\n"
        )
        return
    if args.deployment is None or (args.execute and args.output is None):
        parser.error(
            "--deployment is required; --execute also requires a new --output directory"
        )
    deployment = load_deployment(args.deployment)
    if os.environ.get("ROS_DOMAIN_ID") != "70":
        parser.error(
            "Set explicit ROS_DOMAIN_ID=70; gripper observation uses domain 71"
        )
    from toolkits.realworld_check.fr3_mapping_calibration_ros import RosRuntime

    console = Console()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: setattr(console, "interrupted", True))
    with RosRuntime(deployment, gripper=plan.gripper) as runtime:
        if args.preflight:
            collect_baseline(runtime, deployment)
            result = {"status": "read_only_preflight", "robot_commands_sent": 0}
        else:
            result = execute_case(runtime, plan, args.deployment, args.output, console)
    sys.stdout.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if result.get("status") == "aborted":
        raise SystemExit(2)


if __name__ == "__main__":
    # ROS imports these exception classes by their package-qualified name.
    from toolkits.realworld_check.fr3_mapping_calibration import main as _main

    _main()
