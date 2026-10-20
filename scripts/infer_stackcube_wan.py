# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Single-window StackCube inference with RLinf's action-conditioned Wan pipeline.

Use --check-only first. No reward model or policy is loaded. LIBERO-trained
weights have not been adapted to ManiSkill action semantics.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_sample(index_path, window_id):
    index = json.loads(Path(index_path).read_text())
    if index["backend"] != "wan":
        raise ValueError("Wan inference requires a Wan window index")
    item = index["windows"][window_id]
    t = item["action_start"]
    source = Path(index["source_dir"]) / item["file"]
    with np.load(source, allow_pickle=False) as data:
        condition = data["base_rgb"][t-4:t+1].copy()
        target = data["base_rgb"][t+1:t+9].copy()
        actions = data["actions"][t:t+8].astype(np.float32).copy()
    if condition.shape != (5, 256, 256, 3) or target.shape != (8, 256, 256, 3) or actions.shape != (8, 7):
        raise ValueError("Invalid Wan sample dimensions")
    return condition, target, actions, item


def validate_checkpoint(path):
    path = Path(path)
    files = [path / "model-00001.safetensors", path / "Wan2.2_VAE.pth"]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing Wan checkpoint files: " + ", ".join(missing))
    return files


def run(args):
    import torch
    checkpoint = validate_checkpoint(args.checkpoint)
    condition, targets, actions, item = load_sample(args.index, args.window)
    if args.check_only:
        print(json.dumps({"status": "preflight_ok", "checkpoint": list(map(str, checkpoint)),
                          "window": item, "condition": list(condition.shape),
                          "target": list(targets.shape), "actions": list(actions.shape)}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan inference")
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    print("Loading action-conditioned Wan with CPU offloading enabled...", flush=True)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[ModelConfig(path=str(checkpoint[0]), offload_device="cpu"),
                       ModelConfig(path=str(checkpoint[1]), offload_device="cpu")])
    if not hasattr(pipe, "enable_vram_management"):
        raise RuntimeError("This DiffSynth Wan pipeline has no enable_vram_management()")
    pipe.enable_vram_management()
    # Avoid RLinf backend's unconditional .dit.to(cuda)/.vae.to(cuda).
    # Match backend's conditioning convention (reference + four recent images).
    input_image = [Image.fromarray(condition[0])]
    input_image4 = [[Image.fromarray(frame) for frame in condition[-4:]]]
    history = np.zeros((5, 7), dtype=np.float32)
    action_tensor = torch.from_numpy(np.concatenate([history, actions], axis=0)[None])
    print("Generating eight future frames...", flush=True)
    with torch.inference_mode():
        output = pipe(seed=args.seed, tiled=True, input_image=input_image,
                      input_image4=input_image4, action=action_tensor,
                      height=256, width=256, num_frames=13,
                      num_inference_steps=args.steps, cfg_scale=1.0,
                      batch_size=1)
    frames = output[0][5:]
    if len(frames) != 8:
        raise RuntimeError(f"Expected eight predicted frames, got {len(frames)}")
    prediction = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    mae = float(np.mean(np.abs(prediction.astype(np.float32) - targets.astype(np.float32))))
    result_dir = Path(args.output)
    result_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([condition[-1], targets[-1], prediction[-1]], axis=1)).save(
        result_dir / "wan_comparison.png")
    np.savez_compressed(result_dir / "wan_prediction.npz", prediction=prediction,
                        target=targets, actions=actions)
    (result_dir / "metrics.json").write_text(
        json.dumps({"model": "wan_libero_zero_shot", "base_mae_0_255": mae,
                    "seed": args.seed, "window": item}, indent=2) + "\n")
    print(f"Wan zero-shot base-camera MAE: {mae:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/stackcube_wm_ready/wan_windows.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/wan_libero_spatial"))
    parser.add_argument("--output", type=Path, default=Path("outputs/stackcube_prediction/wan_zero_shot"))
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
