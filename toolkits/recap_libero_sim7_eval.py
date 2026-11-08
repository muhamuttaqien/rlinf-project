"""Evaluate a RECAP CFG pi0.5 checkpoint on LIBERO-10 Task 0.

Usage:
 python -m toolkits.recap_libero_sim7_eval --checkpoint PATH_TO_global_step_3000
Run without --checkpoint for the pretrained pi0.5 baseline.
"""
import argparse
import collections
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from libero.libero import benchmark

from rlinf.models.embodiment.openpi_cfg import get_model
from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict
from toolkits.standalone_eval_scripts.openpi.libero_eval import (
    LIBERO_DUMMY_ACTION, LIBERO_ENV_RESOLUTION, _get_libero_env, _quat2axisangle,
)



def _repair_libero_paths(task, output_dir: Path):
    """Use installed LIBERO resources if ~/.libero/config.yaml points to an old checkout."""
    import yaml
    import libero.libero as libero_paths

    config_path = Path(libero_paths.config_file)
    config = yaml.safe_load(config_path.read_text())
    installed_root = Path(libero_paths.__file__).resolve().parent
    resources = {
        "init_states": ("init_files", task.init_states_file),
        "bddl_files": ("bddl_files", task.bddl_file),
    }
    changed = False
    for key, (subdir, filename) in resources.items():
        location = Path(config[key]) / task.problem_folder / filename
        if location.is_file():
            continue
        installed_location = installed_root / subdir / task.problem_folder / filename
        if not installed_location.is_file():
            raise FileNotFoundError(
                f"LIBERO {key} missing at {location} and {installed_location}. "
                "Install LIBERO's init_files and bddl_files resources."
            )
        config[key] = str(installed_root / subdir)
        changed = True
        print(f"Using installed LIBERO {key}: {config[key]}", flush=True)
    if changed:
        # Avoid changing the user's global ~/.libero/config.yaml.
        output_dir.mkdir(parents=True, exist_ok=True)
        fixed_config = output_dir / "libero_eval_config.yaml"
        fixed_config.write_text(yaml.safe_dump(config))
        libero_paths.config_file = str(fixed_config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Checkpoint global_step_N directory")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-count", type=int, default=3)
    parser.add_argument("--output", default="logs/recap_sim7_eval")
    args = parser.parse_args()
    repo_path = Path(__file__).resolve().parents[1]
    os.environ["REPO_PATH"] = str(repo_path)

    with initialize_config_dir(
        version_base="1.1",
        config_dir=str(repo_path / "examples/offline_rl/config"),
    ):
        cfg = compose(config_name="experiments/libero10_task0/cfg_rl_openpi")
    model = get_model(cfg.actor.model)  # Load pretrained base AND its LIBERO norm stats.
    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser().resolve()
        weights_path = ckpt / "actor/model_state_dict/full_weights.pt"
        if not weights_path.is_file():
            parser.error(f"Checkpoint weights missing: {weights_path}")
        state = as_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint key mismatch; refusing invalid evaluation. "
                f"Missing: {incompatible.missing_keys[:8]}; "
                f"Unexpected: {incompatible.unexpected_keys[:8]}"
            )
        print(f"Loaded RECAP checkpoint: {weights_path}", flush=True)
    else:
        print("Evaluating pretrained pi0.5 baseline", flush=True)

    model = model.to("cuda").eval()
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)
    out_dir = Path(args.output)
    _repair_libero_paths(task, out_dir)
    initial_states = suite.get_task_init_states(0)
    if args.trials > len(initial_states):
        parser.error(f"Requested {args.trials} trials but only {len(initial_states)} init states exist")
    out_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    for episode in range(args.trials):
        env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:
            env.reset()
            obs = env.set_init_state(initial_states[episode])
            action_plan = collections.deque()
            frames = []
            done = False
            for step in range(530):
                if step < 10:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    continue
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                if episode < args.video_count:
                    frames.append(img)
                if not action_plan:
                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"].copy()),
                        obs["robot0_gripper_qpos"],
                    )).astype(np.float32)
                    env_obs = {
                        "main_images": np.asarray([img]),
                        "wrist_images": np.asarray([wrist_img]),
                        "states": np.asarray([state]),
                        "task_descriptions": [str(description)],
                    }
                    actions, _ = model.predict_action_batch(env_obs, mode="eval")
                    action_plan.extend(actions[0, :5])
                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break
            successes += int(done)
            print(f"Episode {episode + 1}/{args.trials}: {'SUCCESS' if done else 'FAIL'} "
                  f"({successes}/{episode + 1}, {100 * successes/(episode + 1):.1f}%)", flush=True)
            if frames:
                suffix = "success" if done else "failure"
                video = out_dir / f"task0_episode{episode + 1}_{suffix}.mp4"
                imageio.mimwrite(video, frames[::10], fps=3)
        finally:
            env.close()
    print(f"LIBERO-10 Task 0 success rate: {successes}/{args.trials} "
          f"= {100 * successes / args.trials:.1f}%", flush=True)
    print(f"Videos: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
