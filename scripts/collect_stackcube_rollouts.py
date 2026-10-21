# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Collect bounded OpenPI StackCube rollouts into a local LeRobot dataset."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers StackCube-v1
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi import get_model

ROOT = Path.cwd()
ACTION_CHUNK = 10
HORIZON = 150
TASK = "Stack the red cube on the green cube"
CONFIG = ROOT / "outputs/stackcube_recap/full_sft/config/resolved_training_config.yaml"
NORM = ROOT / "outputs/stackcube_recap/openpi_norm/stackcube_panda/norm_stats.json"
SFT = ROOT / (
    "outputs/stackcube_recap/full_sft/results/stackcube_panda_pi05_full_sft/"
    "checkpoints/global_step_1084/actor/model_state_dict"
)


def to_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value):
    result = to_numpy(value).reshape(-1)
    if result.size != 1:
        raise ValueError(f"Expected a scalar environment value, found {result.shape}")
    return result.item()


def file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_observation(obs):
    sensors = obs["sensor_data"]
    main = to_numpy(sensors["base_camera"]["rgb"])[..., :3]
    hand = to_numpy(sensors["hand_camera"]["rgb"])[..., :3]
    if main.ndim == 3:
        main = main[None]
    if hand.ndim == 3:
        hand = hand[None]
    state = np.concatenate(
        (to_numpy(obs["agent"]["qpos"]), to_numpy(obs["agent"]["qvel"])), axis=-1
    ).astype(np.float32, copy=False)
    if main.shape != (1, 128, 128, 3) or hand.shape != (1, 128, 128, 3):
        raise ValueError(
            f"Unexpected camera observation shapes {main.shape}, {hand.shape}"
        )
    if state.shape != (1, 18):
        raise ValueError(f"Expected 18D Panda proprioception, got {state.shape}")
    return {
        "main_images": main,
        "wrist_images": hand,
        "states": state,
        "task_descriptions": [TASK],
    }


def pose_record(pose):
    return {
        "position_xyz": to_numpy(pose.p[0]).astype(float).tolist(),
        "quaternion_wxyz": to_numpy(pose.q[0]).astype(float).tolist(),
    }


def controller_targets(env):
    parts = getattr(env.unwrapped.agent.controller, "controllers", {})
    targets = {}
    for name in ("arm", "gripper"):
        part = parts.get(name)
        target = getattr(part, "_target_qpos", None)
        if target is not None:
            targets[name] = to_numpy(target[0]).astype(float).tolist()
    return targets


def scene_signature(env):
    base = env.unwrapped
    return {
        "robot_qpos": to_numpy(base.agent.robot.get_qpos()[0]).astype(float).tolist(),
        "robot_qvel": to_numpy(base.agent.robot.get_qvel()[0]).astype(float).tolist(),
        "cubeA_pose": pose_record(base.cubeA.pose),
        "cubeB_pose": pose_record(base.cubeB.pose),
    }


def select_action_prefix(chunk, count):
    if chunk.ndim != 2 or chunk.shape[1] != 7 or not 1 <= count <= chunk.shape[0]:
        raise ValueError(
            f"Expected (T,7) actions and valid prefix, got {chunk.shape}, {count}"
        )
    return chunk[:count]


def should_stop_after_transition(success, terminated, truncated, step, horizon):
    return success or terminated or truncated or step >= horizon


def sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_features():
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (18,),
            "names": [f"state_{i}" for i in range(18)],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"action_{i}" for i in range(7)],
        },
        "policy_action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"action_{i}" for i in range(7)],
        },
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "is_success": {"dtype": "bool", "shape": (1,), "names": None},
        "terminated": {"dtype": "bool", "shape": (1,), "names": None},
        "truncated": {"dtype": "bool", "shape": (1,), "names": None},
        "grasped": {"dtype": "bool", "shape": (1,), "names": None},
        "lifted_by_2_5cm": {"dtype": "bool", "shape": (1,), "names": None},
        "transported_by_3cm": {"dtype": "bool", "shape": (1,), "names": None},
        "simulator_step": {"dtype": "int64", "shape": (1,), "names": None},
        "inference_call": {"dtype": "int64", "shape": (1,), "names": None},
        "action_index_in_chunk": {"dtype": "int64", "shape": (1,), "names": None},
    }
    for name in ("base_camera", "hand_camera"):
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (3, 128, 128),
            "names": ["channel", "height", "width"],
        }
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--actions-per-inference", type=int, choices=(1, 2, 5, 10), default=5
    )
    parser.add_argument("--horizon", type=int, default=150)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 10:
        parser.error("pilot collection is limited to 1–10 episodes")
    if not 1 <= args.horizon <= HORIZON:
        parser.error(f"horizon must be in [1,{HORIZON}]")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces").mkdir(exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.cuda.set_device(0)
    cfg = OmegaConf.load(CONFIG).actor.model
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.model_path = str(SFT)
    cfg.openpi.task = "eval"
    cfg.openpi.torch_compile = False
    cfg.openpi_data.norm_stats_path = str(NORM)
    model = get_model(cfg).to("cuda:0").eval()
    env = gym.make(
        "StackCube-v1",
        robot_uids="panda_wristcam",
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        num_envs=1,
        sim_backend="physx_cpu",
        render_mode="rgb_array",
        max_episode_steps=args.horizon,
    )
    time_limit = env
    while time_limit is not None and not hasattr(time_limit, "_max_episode_steps"):
        time_limit = getattr(time_limit, "env", None)
    if getattr(time_limit, "_max_episode_steps", None) != args.horizon:
        env.close()
        raise RuntimeError("Gym TimeLimit does not match the requested horizon")
    if not hasattr(env.unwrapped, "evaluate"):
        env.close()
        raise RuntimeError("native StackCube success evaluator is unavailable")

    checkpoint_file = SFT / "full_weights.pt"
    if not checkpoint_file.is_file() or not NORM.is_file():
        env.close()
        raise FileNotFoundError(
            "validated SFT checkpoint or normalization file is missing"
        )
    metadata = {
        "environment": "ManiSkill StackCube-v1",
        "robot": "panda_wristcam",
        "controller": "pd_ee_delta_pose",
        "sim_backend": "physx_cpu",
        "horizon": args.horizon,
        "actions_per_inference": args.actions_per_inference,
        "model_action_chunk_length": ACTION_CHUNK,
        "observation_alignment": "pre-action observation, transition outcome on same LeRobot row",
        "observation": "base_camera RGB, hand_camera RGB, qpos+qvel (18D)",
        "action": "7D policy output sent to env.step; no additional normalization change",
        "task": TASK,
        "seeds": list(range(args.seed, args.seed + args.episodes)),
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": file_sha256(checkpoint_file),
        "norm_stats": str(NORM),
        "norm_stats_sha256": file_sha256(NORM),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "nvml_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "lerobot": __import__("importlib.metadata", fromlist=["version"]).version(
            "lerobot"
        ),
        "flow_matching_rng_seed": "episode_seed * 1000 + inference_call_index",
        "success_signal": "native StackCubeEnv.evaluate().success",
        "stage_flags": "diagnostic heuristics; not native success",
    }
    (args.output / "collection_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    dataset = LeRobotDataset.create(
        repo_id="local/stackcube-recap-rollouts",
        root=args.output / "dataset",
        fps=20,
        robot_type="panda_wristcam",
        features=build_features(),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    summaries = []
    torch.cuda.reset_peak_memory_stats(0)
    try:
        for seed in metadata["seeds"]:
            obs, _ = env.reset(seed=seed)
            base = env.unwrapped
            initial = scene_signature(env)
            signature = sha256_json(initial)
            initial_z = float(to_numpy(base.cubeA.pose.p[0])[2])
            initial_distance = float(
                np.linalg.norm(
                    to_numpy(base.cubeA.pose.p[0])[:2]
                    - to_numpy(base.cubeB.pose.p[0])[:2]
                )
            )
            success_once = grasp_once = lift_once = transport_once = placed_once = False
            terminated = truncated = False
            total_reward = 0.0
            inference_latencies = []
            inference_calls = 0
            start = time.monotonic()
            trace_path = args.output / "traces" / f"seed_{seed}.jsonl"
            with trace_path.open("w") as trace:
                step = 0
                while step < args.horizon and not (
                    terminated or truncated or success_once
                ):
                    policy_obs = policy_observation(obs)
                    rng = torch.Generator(device="cuda:0").manual_seed(
                        seed * 1000 + inference_calls
                    )
                    infer_start = time.monotonic()
                    with torch.inference_mode():
                        predicted, _ = model.predict_action_batch(policy_obs, rng=rng)
                    inference_latencies.append(time.monotonic() - infer_start)
                    if (
                        tuple(predicted.shape) != (1, ACTION_CHUNK, 7)
                        or not torch.isfinite(predicted).all()
                    ):
                        raise ValueError(
                            f"invalid policy chunk {tuple(predicted.shape)} for seed {seed}"
                        )
                    chunk = predicted[0].float().cpu().numpy()
                    inference_calls += 1
                    for chunk_offset, action in enumerate(
                        select_action_prefix(chunk, args.actions_per_inference)
                    ):
                        if step >= args.horizon:
                            break
                        before = obs
                        pre_state = (
                            np.concatenate(
                                (
                                    to_numpy(before["agent"]["qpos"]),
                                    to_numpy(before["agent"]["qvel"]),
                                ),
                                axis=-1,
                            )
                            .reshape(-1)
                            .astype(np.float32)
                        )
                        main = to_numpy(before["sensor_data"]["base_camera"]["rgb"])[
                            0, ..., :3
                        ].astype(np.uint8)
                        hand = to_numpy(before["sensor_data"]["hand_camera"]["rgb"])[
                            0, ..., :3
                        ].astype(np.uint8)
                        executed = action.astype(np.float32, copy=True)
                        obs, reward, term, trunc, info = env.step(executed[None])
                        step += 1
                        terminated, truncated = bool(scalar(term)), bool(scalar(trunc))
                        reward_scalar = float(scalar(reward))
                        total_reward += reward_scalar
                        state_info = info if "success" in info else base.evaluate()
                        success = bool(scalar(state_info.get("success", False)))
                        grasped = bool(
                            scalar(state_info.get("is_cubeA_grasped", False))
                        )
                        placed = bool(
                            scalar(state_info.get("is_cubeA_on_cubeB", False))
                        )
                        red = to_numpy(base.cubeA.pose.p[0]).astype(float)
                        green = to_numpy(base.cubeB.pose.p[0]).astype(float)
                        lifted = float(red[2]) >= initial_z + 0.025
                        transported = (
                            initial_distance
                            - float(np.linalg.norm(red[:2] - green[:2]))
                            >= 0.03
                        )
                        success_once |= success
                        grasp_once |= grasped
                        lift_once |= lifted
                        transport_once |= transported
                        placed_once |= placed
                        frame = {
                            "observation.state": pre_state,
                            "action": executed,
                            "policy_action": action.astype(np.float32, copy=False),
                            "reward": np.asarray([reward_scalar], np.float32),
                            "is_success": np.asarray([success], np.bool_),
                            "terminated": np.asarray([terminated], np.bool_),
                            "truncated": np.asarray([truncated], np.bool_),
                            "grasped": np.asarray([grasped], np.bool_),
                            "lifted_by_2_5cm": np.asarray([lifted], np.bool_),
                            "transported_by_3cm": np.asarray([transported], np.bool_),
                            "simulator_step": np.asarray([step], np.int64),
                            "inference_call": np.asarray([inference_calls], np.int64),
                            "action_index_in_chunk": np.asarray(
                                [chunk_offset], np.int64
                            ),
                            "observation.images.base_camera": main,
                            "observation.images.hand_camera": hand,
                        }
                        dataset.add_frame(frame, task=TASK)
                        row = {
                            "seed": seed,
                            "episode_step": step,
                            "observation_refreshed": chunk_offset == 0,
                            "state_pre_action": pre_state.tolist(),
                            "camera_shapes": [list(main.shape), list(hand.shape)],
                            "tcp_pose_post": pose_record(base.agent.tcp.pose),
                            "cubeA_pose_post": pose_record(base.cubeA.pose),
                            "cubeB_target_pose_post": pose_record(base.cubeB.pose),
                            "gripper_command": float(action[6]),
                            "gripper_qpos_post": to_numpy(
                                base.agent.robot.get_qpos()[0]
                            )[-2:]
                            .astype(float)
                            .tolist(),
                            "policy_action": action.astype(float).tolist(),
                            "executed_action": executed.astype(float).tolist(),
                            "controller_target_qpos": controller_targets(env),
                            "reward": reward_scalar,
                            "terminated": terminated,
                            "truncated": truncated,
                            "native_success": success,
                            "grasped": grasped,
                            "lifted_by_2_5cm": lifted,
                            "transported_by_3cm": transported,
                            "placed": placed,
                            "initial_state_signature": signature,
                        }
                        trace.write(json.dumps(row, separators=(",", ":")) + "\n")
                        if should_stop_after_transition(
                            success, terminated, truncated, step, args.horizon
                        ):
                            break
            dataset.save_episode()
            failure = (
                "success"
                if success_once
                else "no_grasp"
                if not grasp_once
                else "grasped_not_lifted"
                if not lift_once
                else "lifted_not_transported"
                if not transport_once
                else "transported_not_stacked"
                if not placed_once
                else "stacked_not_released_or_stabilized"
            )
            summaries.append(
                {
                    "episode_index": len(summaries),
                    "seed": seed,
                    "initial_state_signature": signature,
                    "steps": step,
                    "success": success_once,
                    "terminated": terminated,
                    "truncated": truncated,
                    "grasped": grasp_once,
                    "lifted_by_2_5cm": lift_once,
                    "transported_by_3cm": transport_once,
                    "placed": placed_once,
                    "total_reward": total_reward,
                    "failure_stage": failure,
                    "inference_calls": inference_calls,
                    "mean_inference_latency_seconds": float(
                        np.mean(inference_latencies)
                    ),
                    "runtime_seconds": time.monotonic() - start,
                    "initial_state": initial,
                    "trace": str(trace_path),
                }
            )
            (args.output / "episode_summaries.json").write_text(
                json.dumps(summaries, indent=2) + "\n"
            )
            print(json.dumps(summaries[-1], sort_keys=True), flush=True)
    finally:
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        env.close()
    metadata["gpu_memory_peak_allocated_mib"] = round(
        torch.cuda.max_memory_allocated(0) / 2**20, 1
    )
    metadata["gpu_memory_peak_reserved_mib"] = round(
        torch.cuda.max_memory_reserved(0) / 2**20, 1
    )
    (args.output / "collection_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "episodes": len(summaries),
                "transitions": sum(x["steps"] for x in summaries),
                "successes": sum(x["success"] for x in summaries),
                "gpu_memory_peak_allocated_mib": metadata[
                    "gpu_memory_peak_allocated_mib"
                ],
                "gpu_memory_peak_reserved_mib": metadata[
                    "gpu_memory_peak_reserved_mib"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
