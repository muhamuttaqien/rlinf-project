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
from contextlib import ExitStack
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np

from toolkits.realworld_check.fr3_live_qualification import read_camera_pair
from toolkits.realworld_check.fr3_multimodal import assemble_fr3_multimodal
from toolkits.realworld_check.fr3_request_live_shadow import (
    make_live_input, prepare_checkpoint, validate_prediction,
)
from toolkits.realworld_check.fr3_request_sft_offline_eval import CONFIG_NAME, make_model_config
from toolkits.realworld_check.franka_observation import BROADCASTER, RosStateSubscriber
from toolkits.realworld_check.fr3_rollout_outcomes import progress

DEFAULT_ROOT = Path("/data1/Muttaqien/fr3_pi05_real_rollouts_20")
EXECUTOR = Path(__file__).with_name("fr3_pi05_ros_executor.py")


class Executor:
    """Read executor stdout independently so ROS keeps spinning during inference."""
    def __init__(self, *, ros_python: str):
        self.q = queue.Queue()
        env = {**os.environ, "ROS_DOMAIN_ID": "70",
               "ROS_AUTOMATIC_DISCOVERY_RANGE": "SUBNET",
               "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp"}
        self.proc = subprocess.Popen(
            [ros_python, "-u", str(EXECUTOR)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            bufsize=1, env=env,
        )
        self.events = []
        self.error = None
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                event = json.loads(line)
                self.events.append(event)
                self.q.put(event)
            except ValueError:
                self.q.put({"event":"fault", "reason":"Malformed ROS executor event"})
        self.q.put({"event":"exited", "reason":"Executor process closed"})

    def send(self, op, **fields):
        if self.proc.poll() is not None:
            raise RuntimeError("ROS executor stopped")
        self.proc.stdin.write(json.dumps({"op": op, **fields}, allow_nan=False) + "\n")
        self.proc.stdin.flush()

    def wait(self, event_name: str, *, timeout=10):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                item = self.q.get(timeout=min(0.1, max(0.01, end-time.monotonic())))
            except queue.Empty:
                continue
            if item.get("event") in ("fault", "exited"):
                raise RuntimeError("ROS executor: " + item.get("reason","unknown fault"))
            if item.get("event") == event_name:
                return item
        raise TimeoutError(f"Executor never returned {event_name}")

    def close(self):
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
    from rlinf.models.embodiment.openpi.transforms.apply import apply_input_transform
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms
    from rlinf.robotics.parts.cameras import Camera, CameraInfo

    ckpt = prepare_checkpoint(args.checkpoint)
    current = progress(args.root)
    episode = current["next_episode"]
    if episode is None:
        raise RuntimeError("All 20 episodes already registered")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU checkpoint inference unavailable")
    if args.max_chunks < 1 or args.max_chunks > 10:
        raise ValueError("max-chunks must be 1..10")
    if args.base_serial == args.wrist_serial:
        raise ValueError("Camera serials must differ")
    episode_dir = args.root / f"episode_{episode:02d}"
    if episode_dir.exists():
        raise FileExistsError(f"Episode folder already exists: {episode_dir}")
    episode_dir.mkdir(parents=True)

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
            args.ros_python, domain_id=70,
            joint_topic=BROADCASTER+"/measured_joint_states",
            pose_topic=BROADCASTER+"/current_pose",
            wrench_topic=BROADCASTER+"/external_wrench_in_base_frame",
        ),
        "gripper": RosStateSubscriber(
            args.ros_python, domain_id=71,
            joint_topic="/fr3_unused_joint", pose_topic="/fr3_unused_pose",
            wrench_topic="/fr3_unused_wrench", gripper_topic="/joint_states",
        ),
    }
    cameras = {
        "base": Camera.of(CameraInfo(name="base", serial_number=args.base_serial,
                                      resolution=(640, 480), fps=30)),
        "wrist": Camera.of(CameraInfo(name="wrist", serial_number=args.wrist_serial,
                                       resolution=(848, 480), fps=30)),
    }
    executor = None
    issued = 0
    print(f"Episode {episode}/20. Ensure SpaceMouse publisher is STOPPED. "
          "Physically supervise with hardware E-stop available.", flush=True)

    with ExitStack() as stack:
        for reader in readers.values():
            reader.start()
            stack.callback(reader.close)
        for camera in cameras.values():
            camera.connect()
            stack.callback(camera.disconnect)
        executor = Executor(ros_python=args.ros_python)
        stack.callback(executor.close)
        executor.wait("ready", timeout=8)
        if input("Type RUN for ONE supervised physical episode: ").strip() != "RUN":
            print("Cancelled; no policy commands.")
            return
        executor.send("begin_episode")
        executor.wait("episode_started")

        with (episode_dir/"policy_command_log.jsonl").open("x") as commands, (
                episode_dir/"measured_state_log.jsonl").open("x") as states:
            for chunk_i in range(args.max_chunks):
                if any(e.get("event") in ("fault", "exited") for e in executor.events):
                    raise RuntimeError("ROS executor fault")
                t0 = time.monotonic()
                frames = read_camera_pair(
                    cameras, max_age_s=0.5, max_skew_s=0.1
                )
                multimodal = assemble_fr3_multimodal(
                    readers["arm"].snapshot(0.5),
                    readers["gripper"].snapshot(0.5),
                    frames, max_camera_receipt_skew_s=0.1,
                )
                raw = make_live_input(multimodal, prompt=args.prompt)
                batch = {
                    "observation/image":raw["observation/image"][None],
                    "observation/wrist_image":raw["observation/wrist_image"][None],
                    "observation/state":raw["observation/state"][None],
                    "prompt":[raw["prompt"]],
                }
                observation_ns = time.monotonic_ns()
                obs = model._observation_to_device(
                    apply_input_transform(transform_in, batch)
                )
                with torch.inference_mode():
                    noise = torch.randn(
                        (1,10,32), generator=rng, device="cuda",
                        dtype=torch.float32
                    )
                    generated = model.sample_actions(obs, num_steps=5, noise=noise)
                    decoded = np.asarray(
                        inverse_actions({"actions": generated.detach().cpu().float().numpy()[0]})["actions"]
                    )[:, :7]
                actions = validate_prediction(decoded)
                file = f"observation_{chunk_i:03d}.npz"
                np.savez_compressed(
                    episode_dir/file,
                    base_rgb=raw["observation/image"],
                    wrist_rgb=raw["observation/wrist_image"],
                    state19=raw["observation/state"],
                )
                # No model request is published until the Jazzy executor
                # independently validates the latest observation and graph.
                issued_ns = time.monotonic_ns()
                before = len(executor.events)
                executor.send("chunk", values=actions.tolist(),
                              observation_ns=observation_ns, issued_ns=issued_ns)
                executor.wait("chunk_done", timeout=4)
                new_events = executor.events[before:]
                action_events = [e for e in new_events if e.get("event")=="action"]
                if len(action_events) != 10:
                    raise RuntimeError("Incomplete physical command chunk")
                for e in action_events:
                    commands.write(json.dumps(e,allow_nan=False)+"\n")
                    states.write(json.dumps({
                        "monotonic_ns":e["monotonic_ns"], "tcp_xyz":e["tcp_xyz"],
                        "observation_file":file, "chunk":chunk_i,
                        "observed_state19": raw["observation/state"].tolist(),
                    },allow_nan=False)+"\n")
                commands.flush()
                states.flush()
                issued += len(action_events)
                print(f"Episode {episode}: {issued} ROS arm commands sent "
                      f"({time.monotonic()-t0:.2f}s/chunk)",flush=True)
            executor.send("end_episode")
            executor.wait("episode_ended")
        print("Robot commands ended. Review peg-in-hole outcome.",flush=True)
        label = input("Outcome [success/failure/aborted]: ").strip().lower()
        if label not in ("success","failure","aborted"):
            raise ValueError("Outcome not recorded: expected success/failure/aborted")
        manifest = {
            "episode_id":episode, "policy_checkpoint":str(ckpt),
            "policy_command_log":"policy_command_log.jsonl",
            "measured_state_log":"measured_state_log.jsonl",
            "outcome":label,"operator_reviewed":True,
            "robot_commands_sent":issued, "shadow_only":False,
            "note":"ROS publish events are not proof of controller acceptance or task success",
        }
        (episode_dir/"outcome.json").write_text(
            json.dumps(manifest,indent=2)+"\n",encoding="utf-8"
        )
        print(json.dumps(progress(args.root),indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--base-model",type=Path,default=Path(
        "/data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_base_rlinf"))
    p.add_argument("--norm-dir",type=Path,default=Path(
        "/data1/Muttaqien/fr3_peg_request_norm_v1"))
    p.add_argument("--base-serial",default="337122075473")
    p.add_argument("--wrist-serial",default="218622277827")
    p.add_argument("--ros-python",default="/usr/bin/python3")
    p.add_argument("--prompt",default="Insert the peg into the hole.")
    p.add_argument("--max-chunks",type=int,default=3)
    p.add_argument("--seed",type=int,default=0)
    a=p.parse_args()
    run(a)


if __name__ == "__main__":
    main()
