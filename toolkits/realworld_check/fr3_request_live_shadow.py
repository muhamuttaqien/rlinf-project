# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Live FR3 pi0.5 REQUEST shadow inference; strictly read-only (NO robot motion).

Uses the qualified dual-camera/ROS state reader, the original 19D request-state
projection, the official OpenPI request input/output transforms and the
consolidated SFT checkpoint. Never opens ROS publishers, action clients,
command topics or control interfaces. Predictions are logged, NOT executed.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np

from toolkits.realworld_check.fr3_live_qualification import read_camera_pair
from toolkits.realworld_check.fr3_multimodal import assemble_fr3_multimodal
from toolkits.realworld_check.fr3_policy_contract import candidate_19d_state
from toolkits.realworld_check.fr3_request_sft_offline_eval import (
    CONFIG_NAME, make_model_config,
)
from toolkits.realworld_check.franka_observation import BROADCASTER, RosStateSubscriber


def make_live_input(multimodal: dict, *, prompt: str) -> dict:
    """Exactly the same logical RGB/19D inputs as request-SFT, no actions."""
    if not prompt.strip():
        raise ValueError("Task prompt cannot be empty")
    s = multimodal["state"]
    state = candidate_19d_state(s)
    views = multimodal["policy_images_rgb"]
    if set(views) != {"base", "wrist"} or state.shape != (19,):
        raise ValueError("Unexpected live input roles/state")
    output = {
        "observation/image": np.ascontiguousarray(views["base"]),
        "observation/wrist_image": np.ascontiguousarray(views["wrist"]),
        "observation/state": state,
        "prompt": prompt,
    }
    for key in ("observation/image", "observation/wrist_image"):
        a = output[key]
        if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
            raise ValueError(f"Invalid {key}")
    return output


def validate_prediction(prediction) -> np.ndarray:
    """Require decoded 10x7 finite requests; do NOT clamp or execute."""
    a = np.asarray(prediction, dtype=np.float64)
    if a.shape != (10, 7) or not np.isfinite(a).all():
        raise ValueError("Expected finite decoded (10,7) request prediction")
    return a


def prepare_checkpoint(path: Path) -> Path:
    ckpt = path.expanduser().resolve()
    if (ckpt.name != "full_weights.pt" or not ckpt.is_file()
            or ckpt.stat().st_size < 1_000_000_000
            or "global_step_3000" not in ckpt.parts):
        raise ValueError("Expected final global_step_3000 actor/model_state_dict/full_weights.pt")
    return ckpt


def run(args) -> dict:
    import torch
    from openpi import transforms
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.models.embodiment.openpi.transforms.apply import (
        apply_input_transform,
    )
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms
    from rlinf.robotics.parts.cameras import Camera, CameraInfo

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the trained 3.35B SFT checkpoint")
    ckpt = prepare_checkpoint(args.checkpoint)
    if args.base_serial == args.wrist_serial:
        raise ValueError("Base and wrist cameras must be different")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    # Match held-out SFT inference, including the training split's norm_stats.
    stats_file = args.norm_dir.expanduser().resolve() / "norm_stats.json"
    if not stats_file.is_file():
        raise FileNotFoundError(f"Missing train-only normalization: {stats_file}")
    base_model = str(args.base_model.expanduser().resolve())
    stats_kwargs = {"norm_stats_path": str(stats_file)}
    input_steps, _ = build_openpi_transforms(
        base_model, CONFIG_NAME, data_kwargs=stats_kwargs,
    )
    input_transform = transforms.compose(input_steps)
    # The general OpenPI output pipeline also tries to unnormalize "state".
    # SFT shadow inference provides only "actions", so apply the same
    # action-only inverse used by fr3_request_sft_offline_eval.py.
    train_cfg = get_openpi_config(
        CONFIG_NAME, model_path=base_model, data_kwargs=stats_kwargs,
    )
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_cfg.norm_stats is None or "actions" not in data_cfg.norm_stats:
        raise ValueError("Missing action normalization from train-only FR3 stats")
    inverse_actions = transforms.Unnormalize(
        {"actions": data_cfg.norm_stats["actions"]},
        use_quantiles=data_cfg.use_quantile_norm,
    )
    model = get_model(make_model_config(ckpt))
    model.eval().to("cuda")
    device = torch.device("cuda")
    rng = torch.Generator(device=device).manual_seed(args.seed)

    readers = {
        "arm": RosStateSubscriber(
            args.ros_python, domain_id=70,
            joint_topic=BROADCASTER + "/measured_joint_states",
            pose_topic=BROADCASTER + "/current_pose",
            wrench_topic=BROADCASTER + "/external_wrench_in_base_frame",
        ),
        "gripper": RosStateSubscriber(
            args.ros_python, domain_id=71,
            joint_topic="/fr3_shadow_unused_arm_joint",
            pose_topic="/fr3_shadow_unused_pose",
            wrench_topic="/fr3_shadow_unused_wrench",
            gripper_topic="/joint_states",
        ),
    }
    cameras = {
        "base": Camera.of(CameraInfo(
            name="base", serial_number=args.base_serial,
            resolution=(640, 480), fps=30,
        )),
        "wrist": Camera.of(CameraInfo(
            name="wrist", serial_number=args.wrist_serial,
            resolution=(848, 480), fps=30,
        )),
    }
    succeeded = 0
    with ExitStack() as stack:
        for reader in readers.values():
            reader.start()
            stack.callback(reader.close)
        for camera in cameras.values():
            camera.connect()
            stack.callback(camera.disconnect)
        time.sleep(1.0)

        with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for i in range(args.samples):
                started = time.monotonic()
                frames = read_camera_pair(
                    cameras, max_age_s=args.max_age_s, max_skew_s=args.max_skew_s
                )
                arm = readers["arm"].snapshot(args.max_age_s)
                gripper = readers["gripper"].snapshot(args.max_age_s)
                multimodal = assemble_fr3_multimodal(
                    arm, gripper, frames,
                    max_camera_receipt_skew_s=args.max_skew_s,
                )
                raw = make_live_input(multimodal, prompt=args.prompt)
                batch = {
                    "observation/image": raw["observation/image"][None, ...],
                    "observation/wrist_image": raw["observation/wrist_image"][None, ...],
                    "observation/state": raw["observation/state"][None, ...],
                    "prompt": [raw["prompt"]],
                }
                obs = model._observation_to_device(
                    apply_input_transform(input_transform, batch)
                )
                with torch.inference_mode():
                    noise = torch.randn(
                        (1, 10, 32), device=device,
                        generator=rng, dtype=torch.float32,
                    )
                    generated = model.sample_actions(
                        obs, num_steps=args.denoising_steps, noise=noise
                    )
                    if tuple(generated.shape) != (1, 10, 32):
                        raise ValueError("Unexpected model action tensor dimensions")
                    raw_actions = generated.detach().cpu().float().numpy()[0]
                    decoded = np.asarray(
                        inverse_actions({"actions": raw_actions})["actions"]
                    )[:, :7]
                prediction = validate_prediction(decoded)

                # Save a local observation snapshot alongside logged predictions
                # for provenance. File contains RGB uint8 and raw measured 19D.
                frame_file = f"observation_{i:03d}.npz"
                np.savez_compressed(
                    output / frame_file,
                    base_rgb=raw["observation/image"],
                    wrist_rgb=raw["observation/wrist_image"],
                    state19=raw["observation/state"],
                )
                record = {
                    "sample": i,
                    "observation_file": frame_file,
                    "prompt": args.prompt,
                    "camera_receipt_skew_s": multimodal["metadata"]["camera_receipt_skew_s"],
                    "camera_receipt_monotonic_ns": {
                        role: frames[role]["received_monotonic_ns"]
                        for role in ("base", "wrist")
                    },
                    "predicted_request_chunk_7d": prediction.tolist(),
                    "prediction_units": "normalized_SpaceMouse_requests_not_TCP_velocity",
                    "controller_accepted_action": None,
                    "executed_action": None,
                    "is_success": None,
                    "robot_commands_sent": 0,
                    "inference_elapsed_s": time.monotonic() - started,
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                succeeded += 1
                print(f"Live shadow prediction {succeeded}/{args.samples} saved", flush=True)
                time.sleep(max(0., 1 / args.fps - (time.monotonic() - started)))

    report = {
        "mode": "live_sensor_shadow_inference_only",
        "model": CONFIG_NAME,
        "checkpoint": str(ckpt),
        "samples": succeeded,
        "observation_sources": "FR3 domain70, Robotiq domain71, local base+wrist RGB",
        "action_horizon": 10,
        "action_dimension": 7,
        "robot_commands_sent": 0,
        "hardware_motion_requested": False,
        "physical_execution_approved": False,
        "recap_rollout_training_ready": False,
        "clock_exposure_sync_verified": False,
        "command_mapping_verified": False,
        "result_file": str(output / "predictions.jsonl"),
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-model", type=Path, default=Path(
        "/data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_base_rlinf"))
    p.add_argument("--norm-dir", type=Path, default=Path(
        "/data1/Muttaqien/fr3_peg_request_norm_v1"))
    p.add_argument("--base-serial", default="337122075473")
    p.add_argument("--wrist-serial", default="218622277827")
    p.add_argument("--ros-python", default="/usr/bin/python3")
    p.add_argument("--prompt", default="Insert the peg into the hole.")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max-age-s", type=float, default=0.5)
    p.add_argument("--max-skew-s", type=float, default=0.1)
    p.add_argument("--denoising-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (not 1 <= args.samples <= 20 or not 0 < args.fps <= 10
            or not 1 <= args.denoising_steps <= 20
            or not 0 < args.max_age_s <= 1 or not 0 < args.max_skew_s <= 0.2):
        p.error("Invalid live shadow bounds (samples 1..20, FPS 0..10)")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
