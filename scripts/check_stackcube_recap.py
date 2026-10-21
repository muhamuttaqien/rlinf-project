# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Validate StackCube RGB replay, LeRobot conversion, and OpenPI batches."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

from rlinf.utils.logging import get_logger

logger = get_logger()


def validate_dataset(replay: Path, dataset: Path) -> tuple[dict, dict]:
    """Compare converted rows with source trajectories and compute raw stats."""
    info = json.loads((dataset / "meta/info.json").read_text())
    tables = [pq.read_table(p) for p in sorted((dataset / "data").rglob("*.parquet"))]
    import pyarrow as pa

    table = pa.concat_tables(tables).sort_by([("index", "ascending")])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    episode_ids = np.asarray(table["episode_index"].to_pylist())
    outcomes = np.asarray(table["is_success"].to_pylist()).reshape(-1)
    if states.shape != (len(table), 18) or actions.shape != (len(table), 7):
        raise ValueError(
            f"Expected state [N,18] and action [N,7], got {states.shape}, {actions.shape}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Expected finite state and action values")
    episodes = []
    with h5py.File(replay, "r") as source:
        names = sorted(source, key=lambda name: int(name.split("_")[-1]))
        if len(names) != info["total_episodes"]:
            raise ValueError("Source and converted episode counts differ")
        for index, name in enumerate(names):
            group = source[name]
            mask = episode_ids == index
            n = len(group["actions"])
            np.testing.assert_array_equal(actions[mask], group["actions"][:])
            state = np.concatenate(
                [group[f"obs/agent/{key}"][:n] for key in ("qpos", "qvel")], axis=-1
            )
            np.testing.assert_array_equal(states[mask], state)
            success = np.asarray(group["success"]).reshape(-1)
            np.testing.assert_array_equal(outcomes[mask], success)
            if not success[-1]:
                raise ValueError(f"Expected successful demonstration: {name}")
            hits = np.flatnonzero(success)
            episodes.append(
                {"episode": index, "steps": n, "first_success_step": int(hits[0] + 1)}
            )
    if len(table) != info["total_frames"]:
        raise ValueError("Converted frame count disagrees with metadata")
    if info["fps"] != 20:
        raise ValueError(f"Expected 20 Hz control, got {info['fps']}")
    stats = {}
    for key, array in (("state", states), ("actions", actions)):
        stats[key] = {
            "mean": array.mean(axis=0).tolist(),
            "std": array.std(axis=0).tolist(),
            "q01": np.quantile(array, 0.01, axis=0).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).tolist(),
        }
    return {"frames": len(table), "fps": info["fps"], "episodes": episodes}, {
        "norm_stats": stats
    }


def validate_videos(dataset: Path, frames: int) -> dict:
    """Decode every stored camera frame and check synchronization counts."""
    import av

    counts = {}
    for camera in ("base_camera", "hand_camera"):
        count = 0
        paths = sorted(
            (dataset / "videos" / f"observation.images.{camera}").rglob("*.mp4")
        )
        for path in paths:
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    if (frame.height, frame.width) != (128, 128):
                        raise ValueError(f"Expected 128x128 RGB: {path}")
                    count += 1
        if count != frames:
            raise ValueError(f"Expected {frames} frames for {camera}, decoded {count}")
        counts[camera] = count
    return counts


def validate_loader(dataset: Path, norm_stats: Path) -> dict:
    """Fetch a real CPU batch through RLinf's official OpenPI SFT loader."""
    import torch
    from hydra import compose, initialize_config_dir
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.data.datasets.openpi import build_openpi_sft_dataloader

    raw_dataset = LeRobotDataset(
        str(dataset.resolve()),
        delta_timestamps={"action": [step / 20 for step in range(10)]},
        video_backend="pyav",
    )
    raw_table = pq.read_table(sorted((dataset / "data").rglob("*.parquet"))).sort_by(
        [("index", "ascending")]
    )
    episode_ids = np.asarray(raw_table["episode_index"].to_pylist())
    raw_actions = np.asarray(raw_table["action"].to_pylist())
    boundary_windows = 0
    for episode in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode)
        for index in (indices[0], indices[-1]):
            sample = raw_dataset[int(index)]
            expected = raw_actions[
                np.minimum(np.arange(index, index + 10), indices[-1])
            ]
            np.testing.assert_allclose(sample["action"].numpy(), expected)
            if sample["task"] != "Stack the red cube on the green cube":
                raise ValueError(f"Unexpected task instruction: {sample['task']}")
            boundary_windows += 1
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        config_dir=str(root / "examples/sft/config"), version_base="1.1"
    ):
        cfg = compose(
            config_name="stackcube_sft_openpi_pi05",
            overrides=[
                f"hydra.searchpath=[file://{root / 'examples/embodiment/config'}]",
                f"actor.model.openpi_data.norm_stats_path={json.dumps(str(norm_stats.resolve()))}",
            ],
        )
    loader, _ = build_openpi_sft_dataloader(cfg, 1, 0, str(dataset), eval_dataset=True)
    observation, actions = next(iter(loader))
    if tuple(actions.shape) != (1, 10, 32) or tuple(observation.state.shape) != (1, 32):
        raise ValueError(
            f"Unexpected batch shapes: {actions.shape}, {observation.state.shape}"
        )
    for tensor in (observation.state, actions, *observation.images.values()):
        if not torch.isfinite(tensor).all():
            raise ValueError("Expected finite OpenPI batch tensors")
    if not observation.image_masks["left_wrist_0_rgb"].all():
        raise ValueError("Expected the wrist image to be present")
    if observation.image_masks["right_wrist_0_rgb"].any():
        raise ValueError("Expected the unused right camera to be masked")
    return {
        "actions": list(actions.shape),
        "state": list(observation.state.shape),
        "images": {k: list(v.shape) for k, v in observation.images.items()},
        "tokens": list(observation.tokenized_prompt.shape),
        "config_name": "pi05_stackcube",
        "model_loaded": False,
        "episode_boundary_windows": boundary_windows,
    }


def validate_environment(norm_stats: Path | None = None) -> dict:
    """Check the Panda RGB interface and truncation at exactly 50 steps."""
    from omegaconf import OmegaConf

    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv

    config_path = (
        Path(__file__).resolve().parents[1]
        / "examples/embodiment/config/env/maniskill_stack_cube.yaml"
    )
    cfg = OmegaConf.load(config_path)
    env = ManiskillEnv(cfg, 1, 0, 1, None)
    try:
        observation, _ = env.reset()
        if env.env.action_space.shape != (7,) or env.env.unwrapped.control_freq != 20:
            raise ValueError("Expected seven-dimensional actions and 20 Hz control")
        if tuple(observation["states"].shape) != (1, 18):
            raise ValueError("Expected [qpos, qvel] with 18 proprioceptive dimensions")
        agent = env.env.unwrapped.get_obs()["agent"]
        np.testing.assert_array_equal(
            observation["states"].cpu().numpy(),
            np.concatenate(
                [agent[key].cpu().numpy() for key in ("qpos", "qvel")], axis=-1
            ),
        )
        for key in ("main_images", "wrist_images"):
            if tuple(observation[key].shape) != (
                1,
                128,
                128,
                3,
            ):
                raise ValueError(f"Unexpected camera shape: {key}")
        if norm_stats is not None:
            from openpi.transforms import compose

            from rlinf.models.embodiment.openpi.transforms.apply import (
                apply_input_transform,
            )
            from rlinf.models.embodiment.openpi.transforms.env import repack_env_obs
            from rlinf.models.embodiment.openpi.transforms.pipeline import (
                build_openpi_transforms,
            )

            inputs, _ = build_openpi_transforms(
                "checkpoints/torch/pi05_base",
                "pi05_stackcube",
                {"norm_stats_path": str(norm_stats.resolve())},
            )
            processed = apply_input_transform(
                compose(inputs),
                repack_env_obs("pi05_stackcube", observation, select_state=lambda x: x),
            )
            if tuple(processed["state"].shape) != (1, 32):
                raise ValueError("Expected the inference state padded to 32 dimensions")
            if not processed["image_mask"]["left_wrist_0_rgb"].all():
                raise ValueError("Expected the inference wrist camera to be present")
        for step in range(1, 51):
            _, _, terminated, truncated, _ = env.step(
                np.zeros((7,), dtype=np.float32), auto_reset=False
            )
            if bool(terminated) or bool(truncated) != (step == 50):
                raise ValueError(
                    f"Expected truncation only at step 50, got step {step}"
                )
        return {
            "robot": "panda_wristcam",
            "control_freq": 20,
            "truncated_at_step": 50,
            "seed": cfg.seed,
            "state_dim": 18,
            "inference_preprocessing_checked": norm_stats is not None,
        }
    finally:
        env.env.close()


def main() -> None:
    """Write validation results and normalization into a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path(
            "outputs/stackcube_recap/raw/StackCube-v1/motionplanning/stackcube_expert.rgb.pd_ee_delta_pose.physx_cpu.h5"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/stackcube_recap/lerobot_stackcube_v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-loader", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output exists: {args.output}; choose a fresh directory")
    report, stats = validate_dataset(args.replay, args.dataset)
    report["video_frames"] = validate_videos(args.dataset, report["frames"])
    report["evaluation_max_steps"] = 50
    report["normalization_scope"] = (
        "all demonstration frames; no held-out dataset claim"
    )
    with args.replay.open("rb") as source:
        report["source_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    args.output.mkdir(parents=True)
    stats_path = args.output / "norm_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    try:
        if args.check_env:
            report["environment"] = validate_environment(
                stats_path if args.check_loader else None
            )
        if args.check_loader:
            report["loader"] = validate_loader(args.dataset, stats_path)
        report["runtime"] = {
            "python": sys.executable,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "lerobot", "mani_skill", "numpy", "pyarrow")
            },
            "module_paths": {
                name: module.__file__
                for name in ("torch", "lerobot", "jax", "openpi")
                if (module := sys.modules.get(name)) is not None
            },
            "training_environment_validated": False,
        }
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    logger.info("StackCube validation passed: %s", args.output)


if __name__ == "__main__":
    main()
