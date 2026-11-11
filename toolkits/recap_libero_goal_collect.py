"""Collect task-specific LIBERO-Goal Task 0 LeRobot data for RECAP.

Creates three disjoint datasets: successful SFT rollouts, mixed policy rollouts,
and held-out mixed evaluation rollouts. This is NOT expert demonstration data.
Actual action, observation, task and success information are recorded at every
simulator step; video-only SIM-7 evaluation artifacts are not sufficient.

Usage:
    python -m toolkits.recap_libero_goal_collect --split all
    python -m toolkits.recap_libero_goal_collect --split sft --sft-successes 48

The collector never silently overwrites a dataset, even after an interruption.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = Path("/data1/Muttaqien/rlinf_data/recap_libero_goal_task0")
DEFAULT_BASELINE = Path("/data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_libero_sft_rlinf")
SPLITS = {
    "sft": ("libero_goal_task0_sft", 10000),
    "train": ("libero_goal_task0_train", 20000),
    "eval": ("libero_goal_task0_eval", 30000),
}


def episode_frames(observations: list[tuple], success: bool, task: str) -> list[dict]:
    """Build LeRobot records from PRE-action observations and executed 7D actions."""
    if not observations:
        raise ValueError("Empty LIBERO rollout")
    frames = []
    for index, (image, wrist, state, action) in enumerate(observations):
        image = np.asarray(image, dtype=np.uint8)
        wrist = np.asarray(wrist, dtype=np.uint8)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if (image.ndim != 3 or image.shape[2] != 3 or wrist.shape != image.shape
                or state.shape != (8,) or action.shape != (7,)
                or not np.isfinite(state).all() or not np.isfinite(action).all()):
            raise ValueError("Invalid LIBERO record: expected RGB (H,W,3), state (8,), action (7,)")
        frames.append({
            "image": image.copy(),
            "wrist_image": wrist.copy(),
            "state": state.copy(),
            "actions": action.copy(),
            "done": np.asarray([index == len(observations) - 1], dtype=np.bool_),
            "is_success": np.asarray([success], dtype=np.bool_),
            "task": task,
        })
    return frames


def features(image_shape: tuple[int, int, int]) -> dict:
    """Explicit LeRobot schema. RECAP requires frame-level episode success."""
    return {
        "image": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
        "is_success": {"dtype": "bool", "shape": (1,), "names": ["is_success"]},
    }


def _make_writer(root: Path, image_shape: tuple[int, int, int], fps: int):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset.create(
        repo_id=root.name,
        root=root,
        robot_type="libero",
        fps=fps,
        features=features(image_shape),
        image_writer_threads=4,
        image_writer_processes=2,
    )


def _rollout(model, task, initial_state, episode_seed: int, max_steps: int) -> tuple[list[tuple], bool, str]:
    from toolkits.recap_libero_sim7_eval import _repair_libero_paths
    from toolkits.standalone_eval_scripts.openpi.libero_eval import (
        LIBERO_DUMMY_ACTION, LIBERO_ENV_RESOLUTION, _get_libero_env, _quat2axisangle,
    )
    # _repair_libero_paths is performed once before the loop by collect().
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    import torch
    torch.manual_seed(episode_seed)
    torch.cuda.manual_seed_all(episode_seed)

    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, episode_seed)
    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        action_plan = deque()
        observations = []
        success = False
        for step in range(max_steps):
            if step < 10:
                obs, _, success, _ = env.step(LIBERO_DUMMY_ACTION)
                continue
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            state = np.concatenate((
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"],
            )).astype(np.float32)
            if not action_plan:
                batch = {
                    "main_images": image[None],
                    "wrist_images": wrist[None],
                    "states": state[None],
                    "task_descriptions": [str(task_description)],
                }
                actions, _ = model.predict_action_batch(batch, mode="eval")
                action_plan.extend(actions[0, :5])
            action = np.asarray(action_plan.popleft(), dtype=np.float32).reshape(-1)
            if action.shape != (7,):
                raise ValueError(f"Policy returned action shape {action.shape}, expected (7,)")
            # Record the actual executed action, never reconstruct it from MP4s.
            observations.append((image, wrist, state, action))
            obs, _, success, _ = env.step(action.tolist())
            if success:
                break
        return observations, bool(success), str(task_description)
    finally:
        env.close()


def collect(args: argparse.Namespace) -> None:
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from hydra import compose, initialize_config_dir
    from libero.libero import benchmark
    from rlinf.models.embodiment.openpi_cfg import get_model
    from rlinf.data.storage.lerobot.compat import add_frame_to_dataset
    from toolkits.recap_libero_sim7_eval import _repair_libero_paths

    repo = Path(__file__).resolve().parents[1]
    output_root = args.output_root.expanduser().resolve()
    checkpoint = args.model_path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Original SFT checkpoint missing: {checkpoint}")
    selections = list(SPLITS) if args.split == "all" else [args.split]
    pending = []
    for split in selections:
        name, seed_offset = SPLITS[split]
        dest = output_root / name
        requested = args.sft_successes if split == "sft" else args.train_episodes if split == "train" else args.eval_episodes
        if dest.exists():
            manifest_file = dest / "collection_manifest.json"
            if manifest_file.is_file():
                prior = json.loads(manifest_file.read_text(encoding="utf-8"))
                if (prior.get("suite") == "libero_goal" and prior.get("task_index") == 0
                        and prior.get("episodes") == requested and prior.get("split") == split
                        and prior.get("baseline_checkpoint") == str(checkpoint)
                        and (dest / "meta/info.json").is_file()):
                    print(f"Reuse completed {split} data: {dest}", flush=True)
                    continue
            raise FileExistsError(
                f"Existing incomplete/incompatible {split} dataset: {dest}. "
                "Preserve it for inspection; choose a new --output-root rather than overwriting."
            )
        pending.append((split, dest, requested, seed_offset))
    if not pending:
        print("All requested task-specific datasets already present.", flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base="1.1", config_dir=str(repo / "examples/offline_rl/config")):
        cfg = compose(config_name="experiments/libero10_task0/cfg_rl_openpi")
    cfg.actor.model.model_path = str(checkpoint)
    cfg.actor.model.openpi.guidance_type = "no_guide"
    model = get_model(cfg.actor.model).to("cuda").eval()
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    _repair_libero_paths(task, output_root)
    init_states = suite.get_task_init_states(0)
    if len(init_states) < 1:
        raise RuntimeError("LIBERO-Goal Task 0 has no initial states")

    try:
        for split, dest, requested, base_seed in pending:
            writer = None
            successes = 0
            saved = 0
            attempts = 0
            limit = max(args.max_sft_attempts, requested) if split == "sft" else requested
            prompt = None
            print(f"Recording {split} ({requested} episodes) into {dest}", flush=True)
            try:
                while saved < requested and attempts < limit:
                    seed = base_seed + attempts
                    init = init_states[attempts % len(init_states)]
                    trajectory, success, task_text = _rollout(model, task, init, seed, args.max_steps)
                    attempts += 1
                    if split == "sft" and not success:
                        print(f"SFT candidate {attempts}: failure (excluded)", flush=True)
                        continue
                    if not trajectory:
                        print(f"{split} candidate {attempts}: empty (excluded)", flush=True)
                        continue
                    if writer is None:
                        writer = _make_writer(dest, trajectory[0][0].shape, args.fps)
                    for frame in episode_frames(trajectory, success, task_text):
                        add_frame_to_dataset(writer, frame)
                    writer.save_episode()
                    successes += int(success)
                    saved += 1
                    prompt = task_text
                    print(f"{split}: saved {saved}/{requested}, success={success}", flush=True)
                if saved != requested:
                    raise RuntimeError(
                        f"Not enough successful SFT rollouts: {saved}/{requested} after {attempts} attempts. "
                        f"Dataset remains incomplete at {dest}; use a fresh output root after reviewing."
                    )
                if split != "sft" and (successes == 0 or successes == requested):
                    raise RuntimeError(
                        f"{split} must contain successful AND failed rollouts; got {successes}/{requested} successes."
                    )
                if writer is not None and getattr(writer, "image_writer", None) is not None:
                    writer.image_writer.wait_until_done()
                if writer is None:
                    raise RuntimeError(f"No episodes recorded for {split}")
                manifest = {
                    "suite": "libero_goal", "task_index": 0, "split": split,
                    "episodes": requested, "successes": successes, "attempts": attempts,
                    "baseline_checkpoint": str(checkpoint), "task_description": prompt,
                    "fps": args.fps, "max_steps": args.max_steps,
                    "seed_start": base_seed,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source": "autonomous SFT policy rollouts (NOT expert demonstrations)",
                }
                manifest_path = dest / "collection_manifest.json"
                tmp = manifest_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                tmp.replace(manifest_path)
            finally:
                if writer is not None and getattr(writer, "image_writer", None) is not None:
                    writer.image_writer.wait_until_done()
    finally:
        del model
    print("LIBERO-Goal Task 0 datasets ready for the preflight.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--sft-successes", type=int, default=48)
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--max-sft-attempts", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=530)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if min(args.sft_successes, args.train_episodes, args.eval_episodes,
           args.max_sft_attempts, args.max_steps, args.fps) < 1:
        parser.error("All episode counts, limits, FPS, and steps must be positive")
    if args.train_episodes < 2 or args.eval_episodes < 2 or args.max_steps <= 10:
        parser.error("Need at least 2 rollout episodes per split and max-steps > 10")
    collect(args)


if __name__ == "__main__":
    main()
