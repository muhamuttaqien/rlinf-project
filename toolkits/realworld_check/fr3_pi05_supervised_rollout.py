# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Collect one supervised physical FR3/Robotiq π0.5 rollout per invocation.

Runs the already-trained 3000-step SFT model in the training venv, and a
separate ROS Jazzy /usr/bin/python3 command process on domain 70. No new
training. The ROS executor exclusively owns control topics and enforces a
bounded measured-pose displacement, 20s duration, and a stale pose watchdog.

Requires a physically present operator at the approved E-stop, a clear
workspace, and SpaceMouse publisher OFF. Console consent is per episode.
Controller neutral request is not a safety-rated stop.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

import numpy as np

from toolkits.realworld_check.fr3_live_qualification import read_camera_pair
from toolkits.realworld_check.fr3_multimodal import assemble_fr3_multimodal
from toolkits.realworld_check.fr3_pi05_request_adapter import load_mapping_verification
from toolkits.realworld_check.fr3_request_live_shadow import (
    make_live_input,
    prepare_checkpoint,
    validate_prediction,
)
from toolkits.realworld_check.fr3_request_sft_offline_eval import (
    CONFIG_NAME,
    make_model_config,
)
from toolkits.realworld_check.fr3_rollout_outcomes import progress
from toolkits.realworld_check.franka_observation import BROADCASTER, RosStateSubscriber

DEFAULT_ROOT = Path("/data1/Muttaqien/fr3_pi05_real_rollouts_20")
EXECUTOR = Path(__file__).with_name("fr3_pi05_ros_executor.py")


class RolloutStopped(RuntimeError):
    """An asynchronous stop revoked permission to submit policy chunks."""


def oldest_observation_ns(multimodal: dict) -> int:
    """Keep the oldest host receipt through acquisition, inference and IPC."""
    metadata = multimodal["metadata"]
    records = [
        metadata["source"][name]
        for name in ("joint_state", "tcp_pose", "external_wrench", "gripper_state")
    ] + [metadata["cameras"][role] for role in ("base", "wrist")]
    stamps = [record["received_monotonic_ns"] for record in records]
    now_ns = time.monotonic_ns()
    if any(type(stamp) is not int or not 0 < stamp <= now_ns for stamp in stamps):
        raise ValueError("Expected positive observation receipts no later than now")
    return min(stamps)


class Executor:
    """Read executor stdout independently so ROS keeps spinning during inference."""

    def __init__(
        self,
        *,
        ros_python: str,
        mapping_mode: str = "legacy_limited",
        mapping_verification: Path | None = None,
        approved_sha256: str | None = None,
    ):
        self.mapping_verification = None
        if mapping_mode == "identity_verified":
            self.mapping_verification = load_mapping_verification(mapping_verification)
            if approved_sha256 != self.mapping_verification["record_sha256"]:
                raise ValueError(
                    "Full-strength execution requires explicit approval of this mapping record"
                )
        elif (
            mapping_mode != "legacy_limited"
            or mapping_verification is not None
            or approved_sha256 is not None
        ):
            raise ValueError(
                "Expected legacy_limited without approval or explicitly approved identity_verified"
            )
        self.event_condition = threading.Condition()
        self.consumed_events = set()
        self.episode_end = None
        self.write_lock = threading.RLock()
        self.stop_sent = False
        self.stopped = threading.Event()
        self.closed = threading.Event()
        self.stop_reason = None
        env = {
            **os.environ,
            "FR3_PI05_MAPPING_MODE": mapping_mode,
            "FR3_PI05_MAPPING_VERIFICATION": str(mapping_verification.resolve())
            if mapping_verification
            else "",
            "FR3_PI05_MAPPING_VERIFICATION_SHA256": approved_sha256 or "",
            "ROS_DOMAIN_ID": "70",
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        }
        self.proc = subprocess.Popen(
            [ros_python, "-u", str(EXECUTOR)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.events = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self.stderr_reader.start()

    def _drain_stderr(self):
        # Keep a full stderr pipe from blocking the independent ROS process.
        while self.proc.stderr.read(4096):
            pass

    def _read(self):
        for line in self.proc.stdout:
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(
                    event.get("event"), str
                ):
                    raise ValueError("Expected an executor event object")
                self._record_event(event)
                if event.get("event") == "fault":
                    self.request_stop("executor_fault")
            except ValueError:
                event = {"event": "fault", "reason": "Malformed ROS executor event"}
                self._record_event(event)
                self.request_stop("executor_fault")
        with self.write_lock:
            self.stop_reason = self.stop_reason or "executor_exit"
            self.stopped.set()
        self._record_event({"event": "exited", "reason": "Executor process closed"})

    def _record_event(self, event: dict) -> None:
        with self.event_condition:
            self.events.append(event)
            if event["event"] == "episode_ended" and self.episode_end is None:
                self.episode_end = event
            self.event_condition.notify_all()

    def send(self, op, **fields):
        with self.write_lock:
            if op in ("chunk", "begin_episode") and (
                self.stopped.is_set() or self.episode_end is not None
            ):
                raise RolloutStopped(self.stop_reason)
            if self.proc.poll() is not None:
                faults = [
                    e.get("reason", "")
                    for e in self.events
                    if e.get("event") == "fault"
                ]
                detail = faults[-1] if faults else f"exit code {self.proc.returncode}"
                raise RuntimeError(f"ROS executor stopped: {detail}")
            self.proc.stdin.write(
                json.dumps({"op": op, **fields}, allow_nan=False) + "\n"
            )
            self.proc.stdin.flush()

    def request_stop(self, reason: str) -> None:
        """Revoke future chunks before asking ROS to cancel the active chunk."""
        with self.write_lock:
            self.stop_reason = self.stop_reason or reason
            self.stopped.set()
            if self.stop_sent or self.episode_end is not None:
                return
            self.stop_sent = True
            try:
                self.send("end_episode")
            except (OSError, RuntimeError):
                pass
        with self.event_condition:
            self.event_condition.notify_all()

    def monitor_stop(
        self, deadline: float, operator_input: TextIO | None = None
    ) -> None:
        """Watch ENTER/EOF and the episode deadline independently of inference."""
        operator_input = sys.stdin if operator_input is None else operator_input

        def monitor():
            while not self.closed.is_set() and not self.stopped.is_set():
                if time.monotonic() >= deadline:
                    self.request_stop("time_limit")
                    return
                try:
                    if select.select([operator_input], [], [], 0.02)[0]:
                        # Any input/EOF requests a stop; never wait for a full line.
                        os.read(operator_input.fileno(), 4096)
                        self.request_stop("operator_stop")
                        return
                except (OSError, ValueError):
                    self.request_stop("operator_input_fault")
                    return

        self.stop_monitor = threading.Thread(target=monitor, daemon=True)
        self.stop_monitor.start()

    def wait(self, event_name: str, *, timeout=10):
        """Consume only the requested event; retain the terminal acknowledgement."""
        end = time.monotonic() + timeout
        with self.event_condition:
            while True:
                for index, item in enumerate(self.events):
                    if item["event"] == "fault" and index not in self.consumed_events:
                        self.consumed_events.add(index)
                        raise RuntimeError(
                            "ROS executor: " + item.get("reason", "unknown fault")
                        )
                if event_name == "episode_ended" and self.episode_end is not None:
                    return self.episode_end
                if event_name != "episode_ended" and (
                    self.stopped.is_set() or self.episode_end is not None
                ):
                    raise RolloutStopped(self.stop_reason or "episode_ended")
                for index, item in enumerate(self.events):
                    if index in self.consumed_events:
                        continue
                    if item["event"] in ("fault", "exited", event_name):
                        self.consumed_events.add(index)
                        if item["event"] in ("fault", "exited"):
                            raise RuntimeError(
                                "ROS executor: " + item.get("reason", "unknown fault")
                            )
                        return item
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Executor never returned {event_name}")
                self.event_condition.wait(timeout=remaining)

    def finish_episode(self, reason: str, *, timeout: float = 10) -> dict:
        """Stop once and await acknowledgement; close the child on failure."""
        self.request_stop(reason)
        try:
            return self.wait("episode_ended", timeout=timeout)
        except (TimeoutError, RuntimeError):
            self.close()
            raise

    def close(self):
        self.closed.set()
        self.stopped.set()
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=2)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        self.reader.join(timeout=2)
        self.stderr_reader.join(timeout=2)
        if hasattr(self, "stop_monitor"):
            self.stop_monitor.join(timeout=1)
        if self.proc.stdin:
            self.proc.stdin.close()
        if self.proc.stdout:
            self.proc.stdout.close()
        if self.proc.stderr:
            self.proc.stderr.close()


def submit_and_archive_chunk(
    executor: Executor,
    values: list[list[float]],
    *,
    observation_ns: int,
    observation_path: Path,
    observation: dict,
) -> dict:
    """Submit fresh actions, then archive their observation during execution.

    An archive failure revokes the active chunk immediately. The caller must
    still wait for chunk completion and retain executor events on every exit.
    """
    issued_ns = time.monotonic_ns()
    executor.send(
        "chunk", values=values, observation_ns=observation_ns, issued_ns=issued_ns
    )
    submitted_ns = time.monotonic_ns()
    try:
        np.savez_compressed(
            observation_path,
            base_rgb=observation["observation/image"],
            wrist_rgb=observation["observation/wrist_image"],
            state19=observation["observation/state"],
        )
    except BaseException:
        executor.request_stop("observation_archive_fault")
        raise
    return {
        "issued_ns": issued_ns,
        "submitted_ns": submitted_ns,
        "archive_done_ns": time.monotonic_ns(),
    }


def archive_action_events(
    action_events: list[dict],
    *,
    commands: TextIO,
    states: TextIO,
    decoded: np.ndarray,
    observation: dict,
    observation_file: str,
    chunk_index: int,
) -> None:
    """Flush every executed action, including a stopped chunk's partial result."""
    for e in action_events:
        commands.write(
            json.dumps(
                {
                    **e,
                    "raw_predicted_7d": decoded[e["command_index"]].tolist(),
                },
                allow_nan=False,
            )
            + "\n"
        )
        states.write(
            json.dumps(
                {
                    "monotonic_ns": e["monotonic_ns"],
                    "tcp_xyz": e["tcp_xyz"],
                    "observation_file": observation_file,
                    "chunk": chunk_index,
                    "observed_state19": observation["observation/state"].tolist(),
                },
                allow_nan=False,
            )
            + "\n"
        )
    commands.flush()
    states.flush()


def record_episode_outcome(episode_dir: Path, manifest: dict) -> None:
    """Create an outcome record without overwriting an existing result."""
    with (episode_dir / "outcome.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


def review_episode_outcome(
    stop_reason: str,
    *,
    execution_error: str | None = None,
    shutdown_error: str | None = None,
) -> tuple[str, bool]:
    """Require operator labels after graceful stops; retain faults as pending aborts."""
    if (
        stop_reason not in ("time_limit", "operator_stop")
        or execution_error is not None
        or shutdown_error is not None
    ):
        return "aborted", False
    try:
        label = input("Outcome [success/failure/aborted]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "aborted", False
    if label not in ("success", "failure", "aborted"):
        raise ValueError("Outcome not recorded: expected success/failure/aborted")
    return label, True


def run(args):
    mode = "identity_verified" if args.verified_request_mapping else "legacy_limited"
    verification = None
    if mode == "identity_verified":
        verification = load_mapping_verification(args.mapping_verification)
    elif args.mapping_verification is not None:
        raise ValueError("--mapping-verification requires --verified-request-mapping")
    import torch
    from openpi import transforms

    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.transforms.apply import apply_input_transform
    from rlinf.models.embodiment.openpi.transforms.pipeline import (
        build_openpi_transforms,
    )
    from rlinf.robotics.parts.cameras import Camera, CameraInfo

    ckpt = prepare_checkpoint(args.checkpoint)
    current = progress(args.root)
    episode = current["next_episode"]
    if episode is None:
        raise RuntimeError(
            "20 usable physical outcomes collected or attempt ID limit reached"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU checkpoint inference unavailable")
    if not 1 <= args.max_episode_seconds <= 15:
        raise ValueError(
            "max-episode-seconds must be within 1..15 for supervised execution"
        )
    if args.base_serial == args.wrist_serial:
        raise ValueError("Camera serials must differ")
    episode_dir = args.root / f"episode_{episode:02d}"
    if episode_dir.exists() and any(episode_dir.iterdir()):
        raise FileExistsError(f"Episode folder contains existing data: {episode_dir}")

    stats_file = args.norm_dir / "norm_stats.json"
    stats_kwargs = {"norm_stats_path": str(stats_file.resolve())}
    transforms_in, _ = build_openpi_transforms(
        str(args.base_model.resolve()), CONFIG_NAME, data_kwargs=stats_kwargs
    )
    transform_in = transforms.compose(transforms_in)
    train_cfg = get_openpi_config(
        CONFIG_NAME, model_path=str(args.base_model.resolve()), data_kwargs=stats_kwargs
    )
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    inverse_actions = transforms.Unnormalize(
        {"actions": data_cfg.norm_stats["actions"]},
        use_quantiles=data_cfg.use_quantile_norm,
    )
    model = get_model(make_model_config(ckpt))
    model.eval().to("cuda")
    rng = torch.Generator(device="cuda").manual_seed(args.seed)

    readers = {
        "arm": RosStateSubscriber(
            args.ros_python,
            domain_id=70,
            joint_topic=BROADCASTER + "/measured_joint_states",
            pose_topic=BROADCASTER + "/current_pose",
            wrench_topic=BROADCASTER + "/external_wrench_in_base_frame",
        ),
        "gripper": RosStateSubscriber(
            args.ros_python,
            domain_id=71,
            joint_topic="/fr3_unused_joint",
            pose_topic="/fr3_unused_pose",
            wrench_topic="/fr3_unused_wrench",
            gripper_topic="/joint_states",
        ),
    }
    cameras = {
        "base": Camera.of(
            CameraInfo(
                name="base",
                serial_number=args.base_serial,
                resolution=(640, 480),
                fps=30,
            )
        ),
        "wrist": Camera.of(
            CameraInfo(
                name="wrist",
                serial_number=args.wrist_serial,
                resolution=(848, 480),
                fps=30,
            )
        ),
    }
    executor = None
    issued = 0
    print(
        f"Attempt {episode} (target 20 usable outcomes). Ensure SpaceMouse publisher is STOPPED. "
        "Physically supervise with hardware E-stop available.",
        flush=True,
    )

    with ExitStack() as stack:
        for reader in readers.values():
            reader.start()
            stack.callback(reader.close)
        for camera in cameras.values():
            camera.connect()
            stack.callback(camera.disconnect)
        approval = (
            f"RUN VERIFIED {verification['record_sha256'][:12]}"
            if verification
            else "RUN"
        )
        print(f"Controller request mapping: {mode}", flush=True)
        if verification:
            print(
                "Operator must match the running controller/configuration and tool frames to "
                f"the physical verification record {verification['record_sha256']}: "
                f"binary={verification['controller_binary_sha256']}, "
                f"config={verification['controller_config_sha256']}, "
                f"frames={verification['base_frame']} -> {verification['tool_frame']}",
                flush=True,
            )
        if (
            input(f"Type {approval} for ONE supervised physical episode: ").strip()
            != approval
        ):
            print("Cancelled; no policy commands.")
            return
        # The ROS executor's live watchdog must not run while the operator
        # is deciding whether to start. Launch it only after confirmation.
        executor = Executor(
            ros_python=args.ros_python,
            mapping_mode=mode,
            mapping_verification=args.mapping_verification,
            approved_sha256=verification["record_sha256"] if verification else None,
        )

        def close_executor_and_save_events():
            try:
                executor.close()
            finally:
                if episode_dir.is_dir():
                    with (episode_dir / "executor_events.jsonl").open("x") as events:
                        for event in executor.events:
                            events.write(json.dumps(event, allow_nan=False) + "\n")

        # Close first so this also preserves partial actions and terminal faults.
        stack.callback(close_executor_and_save_events)
        episode_dir.mkdir(parents=True, exist_ok=True)
        executor.wait("ready", timeout=13)
        executor.send(
            "begin_episode",
            mapping_verification_sha256=verification["record_sha256"]
            if verification
            else None,
        )
        executor.wait("episode_started")
        print(
            "Rollout active: press ENTER to cancel commands, including during inference.",
            flush=True,
        )
        episode_deadline = time.monotonic() + args.max_episode_seconds
        executor.monitor_stop(episode_deadline)
        stop_reason = "time_limit"
        episode_end = None
        execution_error = None
        shutdown_error = None

        with (
            (episode_dir / "policy_command_log.jsonl").open("x") as commands,
            (episode_dir / "measured_state_log.jsonl").open("x") as states,
            (episode_dir / "pipeline_timing.jsonl").open("x") as timings,
        ):
            chunk_i = 0
            while time.monotonic() < episode_deadline:
                if executor.stopped.is_set():
                    stop_reason = executor.stop_reason
                    break
                if any(e.get("event") in ("fault", "exited") for e in executor.events):
                    stop_reason = "executor_fault"
                    executor.request_stop(stop_reason)
                    break
                t0 = time.monotonic()
                acquisition_started_ns = time.monotonic_ns()
                frames = read_camera_pair(cameras, max_age_s=0.5, max_skew_s=0.1)
                multimodal = assemble_fr3_multimodal(
                    readers["arm"].snapshot(0.5),
                    readers["gripper"].snapshot(0.5),
                    frames,
                    max_camera_receipt_skew_s=0.1,
                )
                observation_ns = oldest_observation_ns(multimodal)
                raw = make_live_input(multimodal, prompt=args.prompt)
                observation_ready_ns = time.monotonic_ns()
                batch = {
                    "observation/image": raw["observation/image"][None],
                    "observation/wrist_image": raw["observation/wrist_image"][None],
                    "observation/state": raw["observation/state"][None],
                    "prompt": [raw["prompt"]],
                }
                obs = model._observation_to_device(
                    apply_input_transform(transform_in, batch)
                )
                with torch.inference_mode():
                    noise = torch.randn(
                        (1, 10, 32), generator=rng, device="cuda", dtype=torch.float32
                    )
                    generated = model.sample_actions(obs, num_steps=5, noise=noise)
                    decoded = np.asarray(
                        inverse_actions(
                            {"actions": generated.detach().cpu().float().numpy()[0]}
                        )["actions"]
                    )[:, :7]
                inference_done_ns = time.monotonic_ns()
                if executor.stopped.is_set():
                    stop_reason = executor.stop_reason
                    break
                # Preserve decoded predictions for audit, including NaN/Inf faults.
                from toolkits.realworld_check.fr3_pi05_ros_executor import (
                    validate_chunk,
                )

                try:
                    actions = validate_prediction(decoded)
                    stamp = time.monotonic_ns()
                    actions = np.asarray(
                        validate_chunk(
                            actions.tolist(),
                            observation_ns=observation_ns,
                            issued_ns=stamp,
                            now_ns=stamp,
                        )
                    )
                except (ValueError, TypeError) as exc:
                    stop_reason = "invalid_policy_action"
                    (episode_dir / "invalid_policy_action.json").write_text(
                        json.dumps(
                            {
                                "chunk": chunk_i,
                                "reason": str(exc),
                                "raw_action_7d_repr": repr(decoded.tolist()),
                                "policy_commands_sent_before_fault": issued,
                            },
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    print("Rejected invalid policy action: " + str(exc), flush=True)
                    break
                file = f"observation_{chunk_i:03d}.npz"
                # No model request is published until the Jazzy executor
                # independently validates the latest observation and graph.
                before = len(executor.events)
                timing = {}
                try:
                    timing = submit_and_archive_chunk(
                        executor,
                        actions.tolist(),
                        observation_ns=observation_ns,
                        observation_path=episode_dir / file,
                        observation=raw,
                    )
                    executor.wait("chunk_done", timeout=4)
                except (RolloutStopped, TimeoutError, RuntimeError, OSError) as exc:
                    stop_reason = executor.stop_reason or (
                        "chunk_timeout"
                        if isinstance(exc, TimeoutError)
                        else "executor_fault"
                    )
                    if not isinstance(exc, RolloutStopped):
                        execution_error = str(exc)
                    try:
                        episode_end = executor.finish_episode(stop_reason)
                    except (TimeoutError, RuntimeError) as shutdown_exc:
                        shutdown_error = str(shutdown_exc)
                new_events = executor.events[before:]
                action_events = [e for e in new_events if e.get("event") == "action"]
                timings.write(
                    json.dumps(
                        {
                            "chunk": chunk_i,
                            "observation_ns": observation_ns,
                            "acquisition_started_ns": acquisition_started_ns,
                            "observation_ready_ns": observation_ready_ns,
                            "inference_done_ns": inference_done_ns,
                            **timing,
                            "wait_returned_ns": time.monotonic_ns(),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                timings.flush()
                archive_action_events(
                    action_events,
                    commands=commands,
                    states=states,
                    decoded=decoded,
                    observation=raw,
                    observation_file=file,
                    chunk_index=chunk_i,
                )
                issued += len(action_events)
                if executor.stopped.is_set():
                    stop_reason = executor.stop_reason
                    break
                if len(action_events) != 10:
                    stop_reason = "incomplete_chunk"
                    execution_error = (
                        f"Expected ten action events, found {len(action_events)}"
                    )
                    break
                print(
                    f"Episode {episode}: {issued} policy actions sent (100 Hz arm refresh) "
                    f"({time.monotonic() - t0:.2f}s/chunk)",
                    flush=True,
                )
                chunk_i += 1
            if episode_end is None and shutdown_error is None:
                try:
                    episode_end = executor.finish_episode(stop_reason)
                except (TimeoutError, RuntimeError) as exc:
                    shutdown_error = str(exc)
            executor.stop_monitor.join(timeout=1)
        episode_end = episode_end or executor.episode_end
        print(
            f"Policy rollout ended ({stop_reason}). Review peg-in-hole outcome.",
            flush=True,
        )
        label, operator_reviewed = review_episode_outcome(
            stop_reason, execution_error=execution_error, shutdown_error=shutdown_error
        )
        if not operator_reviewed:
            print(
                "Automatically labeled aborted; operator outcome review is pending.",
                flush=True,
            )
        manifest = {
            "episode_id": episode,
            "policy_checkpoint": str(ckpt),
            "policy_command_log": "policy_command_log.jsonl",
            "measured_state_log": "measured_state_log.jsonl",
            "outcome": label,
            "operator_reviewed": operator_reviewed,
            "execution_interface_qualified": verification is not None,
            "robot_commands_sent": issued,
            "shadow_only": False,
            "policy_actions_sent": issued,
            "arm_publications": episode_end["arm_publications"]
            if episode_end
            else None,
            "max_publish_gap_ns": episode_end["max_publish_gap_ns"]
            if episode_end
            else None,
            "shutdown_confirmed": episode_end is not None,
            "shutdown_error": shutdown_error,
            "execution_error": execution_error,
            "executor_event_log": "executor_events.jsonl",
            "pipeline_timing_log": "pipeline_timing.jsonl",
            "mapping_verification_sha256": verification["record_sha256"]
            if verification
            else None,
            "mapping_verification_record": str(args.mapping_verification.resolve())
            if verification
            else None,
            "stop_reason": stop_reason,
            "control_mode": "supervised_closed_loop_pi05",
            "request_mapping_mode": mode,
            "note": "ROS publish events are not proof of controller acceptance or task success",
        }
        record_episode_outcome(episode_dir, manifest)
        print(json.dumps(progress(args.root), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--base-model",
        type=Path,
        default=Path("/data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_base_rlinf"),
    )
    p.add_argument(
        "--norm-dir",
        type=Path,
        default=Path("/data1/Muttaqien/fr3_peg_request_norm_v1"),
    )
    p.add_argument("--base-serial", default="337122075473")
    p.add_argument("--wrist-serial", default="218622277827")
    p.add_argument("--ros-python", default="/usr/bin/python3")
    p.add_argument("--prompt", default="Insert the peg into the hole.")
    p.add_argument(
        "--max-episode-seconds",
        type=float,
        default=15.0,
        help="Time limit in seconds, maximum 15; press ENTER to stop sooner",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--verified-request-mapping",
        action="store_true",
        help="Match SpaceMouse at gain 1.0 after physical qualification and per-run consent; default is attenuated legacy diagnostics",
    )
    p.add_argument(
        "--mapping-verification",
        type=Path,
        help="Operator-reviewed physical mapping JSON and hashed evidence; required for identity mapping",
    )
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
