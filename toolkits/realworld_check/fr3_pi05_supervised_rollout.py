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
import queue
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

    def __init__(self, *, ros_python: str, mapping_mode: str = "legacy_limited"):
        self.q = queue.Queue()
        self.write_lock = threading.Lock()
        self.stop_lock = threading.Lock()
        self.stopped = threading.Event()
        self.closed = threading.Event()
        self.stop_reason = None
        env = {
            **os.environ,
            "FR3_PI05_MAPPING_MODE": mapping_mode,
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
                self.events.append(event)
                self.q.put(event)
                if event.get("event") == "fault":
                    self.stop_reason = "executor_fault"
                    self.stopped.set()
            except ValueError:
                event = {"event": "fault", "reason": "Malformed ROS executor event"}
                self.events.append(event)
                self.q.put(event)
                self.request_stop("executor_fault")
        self.stop_reason = self.stop_reason or "executor_exit"
        self.stopped.set()
        self.q.put({"event": "exited", "reason": "Executor process closed"})

    def send(self, op, **fields):
        with self.write_lock:
            if op in ("chunk", "begin_episode") and self.stopped.is_set():
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
        with self.stop_lock:
            if self.stopped.is_set():
                return
            self.stop_reason = reason
            self.stopped.set()
            try:
                self.send("end_episode")
            except (OSError, RuntimeError):
                pass

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
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.stopped.is_set() and event_name != "episode_ended":
                raise RolloutStopped(self.stop_reason)
            try:
                item = self.q.get(timeout=min(0.1, max(0.01, end - time.monotonic())))
            except queue.Empty:
                continue
            if item.get("event") in ("fault", "exited"):
                raise RuntimeError(
                    "ROS executor: " + item.get("reason", "unknown fault")
                )
            if item.get("event") == event_name:
                return item
        raise TimeoutError(f"Executor never returned {event_name}")

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


def run(args):
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
        approval = "RUN VERIFIED" if args.verified_request_mapping else "RUN"
        mode = (
            "identity_verified" if args.verified_request_mapping else "legacy_limited"
        )
        print(f"Controller request mapping: {mode}", flush=True)
        if (
            input(f"Type {approval} for ONE supervised physical episode: ").strip()
            != approval
        ):
            print("Cancelled; no policy commands.")
            return
        # The ROS executor's live watchdog must not run while the operator
        # is deciding whether to start. Launch it only after confirmation.
        executor = Executor(ros_python=args.ros_python, mapping_mode=mode)

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
        executor.send("begin_episode")
        executor.wait("episode_started")
        print(
            "Rollout active: press ENTER to cancel commands, including during inference.",
            flush=True,
        )
        episode_deadline = time.monotonic() + args.max_episode_seconds
        executor.monitor_stop(episode_deadline)
        stop_reason = "time_limit"
        episode_end = None

        with (
            (episode_dir / "policy_command_log.jsonl").open("x") as commands,
            (episode_dir / "measured_state_log.jsonl").open("x") as states,
        ):
            chunk_i = 0
            while time.monotonic() < episode_deadline:
                if executor.stopped.is_set():
                    stop_reason = executor.stop_reason
                    break
                if any(e.get("event") in ("fault", "exited") for e in executor.events):
                    raise RuntimeError("ROS executor fault")
                t0 = time.monotonic()
                frames = read_camera_pair(cameras, max_age_s=0.5, max_skew_s=0.1)
                multimodal = assemble_fr3_multimodal(
                    readers["arm"].snapshot(0.5),
                    readers["gripper"].snapshot(0.5),
                    frames,
                    max_camera_receipt_skew_s=0.1,
                )
                observation_ns = oldest_observation_ns(multimodal)
                raw = make_live_input(multimodal, prompt=args.prompt)
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
                if executor.stopped.is_set():
                    stop_reason = executor.stop_reason
                    break
                # Reject nonfinite or out-of-domain predictions before ROS.
                # Save unmodified prediction text even for NaN/Inf, with no clipping.
                from toolkits.realworld_check.fr3_pi05_ros_executor import (
                    validate_chunk,
                )

                try:
                    actions = validate_prediction(decoded)
                    stamp = time.monotonic_ns()
                    validate_chunk(
                        actions.tolist(),
                        observation_ns=observation_ns,
                        issued_ns=stamp,
                        now_ns=stamp,
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
                np.savez_compressed(
                    episode_dir / file,
                    base_rgb=raw["observation/image"],
                    wrist_rgb=raw["observation/wrist_image"],
                    state19=raw["observation/state"],
                )
                # No model request is published until the Jazzy executor
                # independently validates the latest observation and graph.
                issued_ns = time.monotonic_ns()
                before = len(executor.events)
                try:
                    executor.send(
                        "chunk",
                        values=actions.tolist(),
                        observation_ns=observation_ns,
                        issued_ns=issued_ns,
                    )
                    executor.wait("chunk_done", timeout=4)
                except RolloutStopped:
                    stop_reason = executor.stop_reason
                    episode_end = executor.wait("episode_ended")
                new_events = executor.events[before:]
                action_events = [e for e in new_events if e.get("event") == "action"]
                for e in action_events:
                    commands.write(json.dumps(e, allow_nan=False) + "\n")
                    states.write(
                        json.dumps(
                            {
                                "monotonic_ns": e["monotonic_ns"],
                                "tcp_xyz": e["tcp_xyz"],
                                "observation_file": file,
                                "chunk": chunk_i,
                                "observed_state19": raw["observation/state"].tolist(),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                commands.flush()
                states.flush()
                issued += len(action_events)
                if executor.stopped.is_set():
                    stop_reason = executor.stop_reason
                    break
                if len(action_events) != 10:
                    raise RuntimeError("Incomplete physical command chunk")
                print(
                    f"Episode {episode}: {issued} policy actions sent (100 Hz arm refresh) "
                    f"({time.monotonic() - t0:.2f}s/chunk)",
                    flush=True,
                )
                chunk_i += 1
            executor.request_stop(stop_reason)
            if episode_end is None:
                episode_end = executor.wait("episode_ended")
            executor.stop_monitor.join(timeout=1)
        print(
            f"Policy rollout ended ({stop_reason}). Review peg-in-hole outcome.",
            flush=True,
        )
        if stop_reason == "invalid_policy_action":
            label = "aborted"
            print(
                "Automatically labeled aborted due to invalid policy action.",
                flush=True,
            )
        else:
            label = input("Outcome [success/failure/aborted]: ").strip().lower()
            if label not in ("success", "failure", "aborted"):
                raise ValueError(
                    "Outcome not recorded: expected success/failure/aborted"
                )
        manifest = {
            "episode_id": episode,
            "policy_checkpoint": str(ckpt),
            "policy_command_log": "policy_command_log.jsonl",
            "measured_state_log": "measured_state_log.jsonl",
            "outcome": label,
            "operator_reviewed": True,
            "robot_commands_sent": issued,
            "shadow_only": False,
            "policy_actions_sent": issued,
            "arm_publications": episode_end["arm_publications"],
            "max_publish_gap_ns": episode_end["max_publish_gap_ns"],
            "executor_event_log": "executor_events.jsonl",
            "stop_reason": stop_reason,
            "control_mode": "supervised_closed_loop_pi05",
            "request_mapping_mode": mode,
            "note": "ROS publish events are not proof of controller acceptance or task success",
        }
        (episode_dir / "outcome.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
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
        help="Use identity arm request mapping only after separate physical controller/frames/timing verification",
    )
    a = p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
