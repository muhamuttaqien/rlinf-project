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

"""Collect successful Franka Panda StackCube-v1 demonstrations for RLinf RECAP.

Actions use ManiSkill's ``pd_joint_pos`` controller: seven arm-joint targets
and one gripper command. Do not combine this dataset with ``pd_ee_delta_pose``
rollouts (7 actions) or ``pd_joint_delta_pos`` rollouts (8 different semantics).

Run from the RLinf repository root:
    python toolkits/lerobot/collect_maniskill_stackcube_recap.py \
        --repo-id local/stackcube_sft --num-episodes 10
"""

from __future__ import annotations

import argparse
import importlib
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)
TASK = "stack the red cube on top of the green cube and release it"
STATE_DIM = 9
ACTION_DIM = 8
SOLVER_MODULE = "mani_skill.examples.motionplanning.panda.solutions.stack_cube"


def _to_numpy(value: Any) -> np.ndarray:
    """Copy a single-environment tensor or array to CPU."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim and array.shape[0] == 1:
        array = array[0]
    return np.array(array, copy=True)


def _extract_observation(
    obs: dict[str, Any], *, require_wrist_camera: bool = False
) -> dict[str, np.ndarray]:
    """Read the RGB and proprioception fields used by the OpenPI adapter."""
    qpos = _to_numpy(obs["agent"]["qpos"]).astype(np.float32)
    if qpos.shape != (STATE_DIM,) or not np.isfinite(qpos).all():
        raise ValueError(f"Expected finite Panda qpos [{STATE_DIM}], got {qpos.shape}")

    cameras = obs["sensor_data"]

    def rgb(*names: str) -> np.ndarray:
        for name in names:
            if name in cameras and "rgb" in cameras[name]:
                image = _to_numpy(cameras[name]["rgb"])
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(f"Invalid RGB image from {name}: {image.shape}, {image.dtype}")
                return image
        raise ValueError(f"Missing RGB camera {names}; available: {list(cameras)}")

    base_rgb = rgb("base_camera", "3rd_view_camera")
    has_wrist = any(k in cameras for k in ("hand_camera", "wide_hand_camera"))
    if require_wrist_camera and not has_wrist:
        raise ValueError("Two-camera collection requires real hand_camera sensor")
    wrist_rgb = (
        rgb("hand_camera", "wide_hand_camera") if has_wrist else base_rgb
    )
    if require_wrist_camera and np.array_equal(base_rgb, wrist_rgb):
        raise ValueError("Base and wrist camera images are identical")
    return {"image": base_rgb, "wrist_image": wrist_rgb, "state": qpos}


def _solver_succeeded(result: Any) -> bool:
    """Read the environment's actual success flag, never planner status alone."""
    if isinstance(result, dict):
        success = result.get("success", False)
    elif isinstance(result, tuple) and result and isinstance(result[-1], dict):
        success = result[-1].get("success", False)
    else:
        return False
    return bool(_to_numpy(success).reshape(-1)[0])


def _collect_episode(
    env: Any, solve: Any, seed: int, *, require_wrist_camera: bool = False
) -> list[dict[str, Any]] | None:
    """Record every solver step, including calls to the unwrapped ManiSkill env."""
    observations: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []

    original_reset = env.reset
    base_env = env.unwrapped
    original_step = base_env.step

    def reset_hook(*args: Any, **kwargs: Any):
        output = original_reset(*args, **kwargs)
        observations.clear()
        actions.clear()
        observations.append(_extract_observation(
            output[0], require_wrist_camera=require_wrist_camera
        ))
        return output

    def step_hook(action: Any, *args: Any, **kwargs: Any):
        # ManiSkill's solver unwraps the Gym wrapper after resetting.
        parsed = _to_numpy(action).astype(np.float32).reshape(-1)
        if parsed.shape != (ACTION_DIM,) or not np.isfinite(parsed).all():
            raise ValueError(f"Expected finite pd_joint_pos action [{ACTION_DIM}], got {parsed.shape}")
        output = original_step(action, *args, **kwargs)
        actions.append(parsed)
        observations.append(_extract_observation(output[0]))
        return output

    env.reset = reset_hook
    base_env.step = step_hook
    try:
        result = solve(env, seed=seed, debug=False, vis=False)
    finally:
        env.reset = original_reset
        base_env.step = original_step

    if not _solver_succeeded(result):
        return None
    if not actions or len(observations) != len(actions) + 1:
        raise RuntimeError(
            f"Solver returned success but recorded {len(observations)} observations "
            f"and {len(actions)} actions. Check the ManiSkill solver step hook."
        )

    return [
        {
            **observation,
            "actions": action,
            "is_success": np.array([True], dtype=np.bool_),
            "task": TASK,
        }
        for observation, action in zip(observations[:-1], actions)
    ]


def _dataset_path(repo_id: str) -> Path:
    """Resolve a LeRobot repo ID to its local cache path."""
    try:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    path = Path(repo_id).expanduser()
    return path if path.is_absolute() else Path(HF_LEROBOT_HOME) / repo_id


def _dataset_features(first_frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Declare LeRobot's vector and shape-(1,) scalar features."""
    return {
        "image": {
            "dtype": "image",
            "shape": first_frame["image"].shape,
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": first_frame["wrist_image"].shape,
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["actions"],
        },
        "is_success": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["is_success"],
        },
    }


def _create_dataset(repo_id: str, first_frame: dict[str, Any], fps: int):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features=_dataset_features(first_frame),
        image_writer_threads=2,
        image_writer_processes=0,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {value}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/stackcube_sft")
    parser.add_argument("--num-episodes", type=_positive_int, default=10)
    parser.add_argument("--max-attempts", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--image-size", type=_positive_int, default=224)
    parser.add_argument("--control-freq", type=_positive_int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--easy-two-cam", action="store_true",
        help="Collect narrow randomized StackCube with panda_wristcam and real hand camera",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Keep simulator dependencies optional outside this collector.
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    from rlinf.data.storage.lerobot import add_frame_to_dataset

    solve = importlib.import_module(SOLVER_MODULE).solve
    destination = _dataset_path(args.repo_id)
    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")
        shutil.rmtree(destination)

    if args.easy_two_cam:
        import toolkits.lerobot.easy_stackcube_env  # noqa: F401 (register gym env)
    env = gym.make(
        "StackCubeEasyTwoCam-v1" if args.easy_two_cam else "StackCube-v1",
        robot_uids="panda_wristcam" if args.easy_two_cam else "panda",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend=args.sim_backend,
        sim_config={"sim_freq": args.control_freq * 10, "control_freq": args.control_freq},
        sensor_configs={"width": args.image_size, "height": args.image_size},
        max_episode_steps=300,
    )
    dataset = None
    saved = 0
    try:
        for attempt in range(args.max_attempts):
            seed = args.seed + attempt
            try:
                frames = _collect_episode(
                    env, solve, seed, require_wrist_camera=args.easy_two_cam
                )
            except (ValueError, RuntimeError, AssertionError) as exc:
                LOG.warning("Solver attempt seed=%s failed: %s", seed, exc)
                continue
            if frames is None:
                LOG.info("Solver attempt seed=%s did not reach StackCube success", seed)
                continue
            if dataset is None:
                dataset = _create_dataset(args.repo_id, frames[0], args.control_freq)
            for frame in frames:
                add_frame_to_dataset(dataset, frame)
            dataset.save_episode()
            saved += 1
            LOG.info("Saved episode %s/%s (seed=%s, steps=%s)", saved, args.num_episodes, seed, len(frames))
            if saved == args.num_episodes:
                break
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        env.close()

    if saved != args.num_episodes:
        raise RuntimeError(
            f"Collected {saved}/{args.num_episodes} successful episodes after "
            f"{args.max_attempts} attempts; check ManiSkill solver and camera settings"
        )
    LOG.info("LeRobot SFT dataset saved at %s", destination)


if __name__ == "__main__":
    main()
