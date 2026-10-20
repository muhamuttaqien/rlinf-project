# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Launch the upstream OpenVLA-OFT StackCube LoRA SFT smoke test.

No custom trainer, no changes to Wan. Run --check-only first; the official
bfloat16 training recipe needs roughly 26 GB GPU memory for batch size 1.\nUse --qlora for the experimental 16 GB path after applying its upstream patch.
"""
import argparse
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


MIN_VRAM_GIB = 26.0


def validate_paths(root, data_root):
    root = Path(root).resolve()
    data_root = Path(data_root).resolve()
    trainer = root / "vla-scripts" / "finetune.py"
    dataset = data_root / "stack_cube_builder" / "1.0.0"
    if not trainer.is_file():
        raise FileNotFoundError(f"Missing upstream trainer: {trainer}")
    if not (dataset / "dataset_info.json").is_file():
        raise FileNotFoundError(f"Missing StackCube TFDS dataset: {dataset}")
    registry = root / "prismatic/vla/datasets/rlds/oxe/configs.py"
    if not registry.is_file() or '"stack_cube_builder"' not in registry.read_text():
        raise RuntimeError("Run scripts/register_stackcube_openvlaoft.py first")
    return root, data_root


def gpu_status():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"device": None, "vram_gib": 0.0}
        p = torch.cuda.get_device_properties(0)
        return {"device": p.name, "vram_gib": round(p.total_memory / 1024**3, 2)}
    except ImportError:
        return {"device": None, "vram_gib": 0.0}


def training_command(root, data_root, model, output, steps, lora_rank, qlora=False):
    if steps < 1 or lora_rank < 1:
        raise ValueError("Steps and LoRA rank must be positive")
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", "--nproc_per_node=1", "vla-scripts/finetune.py",
        "--vla_path", str(model),
        "--data_root_dir", str(data_root),
        "--dataset_name", "stack_cube_builder",
        "--run_root_dir", str(Path(output).resolve()),
        "--use_l1_regression", "True",
        "--use_diffusion", "False",
        "--use_film", "False",
        "--num_images_in_input", "1" if qlora else "2",
        "--use_proprio", "False",
        "--batch_size", "1",
        "--grad_accumulation_steps", "1",
        "--learning_rate", "5e-4",
        "--max_steps", str(steps),
        "--save_freq", "1",
        "--save_latest_checkpoint_only", "True",
        "--merge_lora_during_training", "False",
        "--image_aug", "False",
        "--shuffle_buffer_size", "128",
        "--lora_rank", str(lora_rank),
        "--wandb_entity", "local",
        "--wandb_project", "stackcube-openvlaoft",
        "--run_id_note", "stackcube_qlora_smoke" if qlora else "stackcube_sft_smoke",
    ]
    if qlora:
        command.extend(["--use_qlora", "True", "--gradient_checkpointing", "True"])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openvla-oft", type=Path,
                        default=Path("external/openvla-oft"))
    parser.add_argument("--data-root", type=Path, default=Path("data/stackcube_rlds"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/stackcube_policy_sft"))
    parser.add_argument("--model", default="openvla/openvla-7b",
                        help="Upstream-supported OpenVLA base model or local directory")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--qlora", action="store_true",
                        help="Use NF4 4-bit QLoRA, gradient checkpointing, one camera")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-low-vram", action="store_true",
                        help="Attempt below 26 GiB (likely CUDA OOM)")
    args = parser.parse_args()
    root, data_root = validate_paths(args.openvla_oft, args.data_root)
    command = training_command(root, data_root, args.model,
                               args.output, args.steps, args.lora_rank, args.qlora)
    patched = "use_qlora: bool = False" in (root / "vla-scripts/finetune.py").read_text()
    dependencies = ["bitsandbytes", "peft", "transformers", "accelerate",
                    "timm", "draccus", "wandb"]
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if args.qlora and not patched and not args.check_only:
        parser.error("Run python scripts/patch_stackcube_openvlaoft_qlora.py first")
    gpu = gpu_status()
    enough = gpu["vram_gib"] >= MIN_VRAM_GIB
    disk = shutil.disk_usage(root).free / 1024**3
    if args.check_only:
        print(json.dumps({
            "dataset": str(data_root / "stack_cube_builder" / "1.0.0"),
            "model": args.model, "gpu": gpu,
            "minimum_recommended_vram_gib": MIN_VRAM_GIB,
            "enough_vram_for_official_bf16_recipe": enough,
            "disk_free_gib": round(disk, 1),
            "training_mode": "4-bit QLoRA (experimental)" if args.qlora else "bf16 LoRA",
            "qlora_patch_installed": patched,
            "missing_python_modules": missing,
            "next": ("4-bit training can be attempted; 16 GiB is not guaranteed to fit"
                     if args.qlora and patched and not missing else
                     "Install the missing modules or apply the QLoRA patch"
                     if args.qlora else
                     "Ready to attempt standard LoRA" if enough else
                     "Use --qlora or a GPU with at least ~26 GiB"),
        }, indent=2))
        return
    if args.qlora and missing:
        parser.error("Missing QLoRA dependencies: " + ", ".join(missing))
    if not args.qlora and not enough and not args.allow_low_vram:
        parser.error(
            f"GPU has {gpu['vram_gib']} GiB, but the official OpenVLA-OFT "
            f"LoRA recipe needs about {MIN_VRAM_GIB} GiB. No training started. "
            "Use a larger GPU; --allow-low-vram only bypasses this check."
        )
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["TOKENIZERS_PARALLELISM"] = "false"
    print("Starting upstream OpenVLA-OFT "
          + ("4-bit QLoRA" if args.qlora else "bf16 LoRA")
          + " SFT (not GRPO).", flush=True)
    print("Command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
