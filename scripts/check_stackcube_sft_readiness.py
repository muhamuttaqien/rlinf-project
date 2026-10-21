# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Preflight π0.5 checkpoint layout and GPU capacity without loading weights."""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Directory containing RLinf model.safetensors (optional for GPU-only check)",
    )
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gib = props.total_memory / (1024**3)
            print(f"GPU {idx}: {props.name}, total VRAM={total_gib:.2f} GiB")
            with torch.cuda.device(idx):
                free, total = torch.cuda.mem_get_info()
            print(f"  Available VRAM={free/(1024**3):.2f} GiB")
            if total_gib < 32:
                print(
                    "  CAUTION: full π0.5 SFT with optimizer states may exceed VRAM. "
                    "This test does not establish feasibility."
                )
    else:
        print("WARNING: GPU unavailable. This check does not run training.")

    if args.checkpoint is None:
        print("No checkpoint supplied: hardware inspection only.")
        return

    path = args.checkpoint.expanduser()
    weights = path if path.is_file() else path / "model.safetensors"
    if weights.suffix != ".safetensors" or not weights.is_file():
        raise SystemExit(
            f"Checkpoint missing: {weights}. RLinf π0.5 base expects "
            "a directory with model.safetensors in the supported layout. "
            "See docs/STACKCUBE_RECAP.md for official conversion instructions."
        )

    print(f"Checkpoint: {weights.resolve()}")
    print(f"Checkpoint size: {weights.stat().st_size / (1024**3):.2f} GiB")
    with safe_open(weights, framework="pt", device="cpu") as f:
        names = list(f.keys())
        print(f"Tensors: {len(names)}")
        bare = any(name.startswith(("llm.", "img.")) for name in names)
        legacy = any(name.startswith("paligemma_with_expert.") for name in names)
        if legacy:
            print(
                "Legacy OpenPI PyTorch key names found. RLinf may convert them "
                "during load, but for predictable reuse convert the checkpoint "
                "to RLinf's bare-Pi0 layout first."
            )
        if not bare and not legacy:
            raise SystemExit(
                "Unrecognized safetensors key prefixes; expected bare Pi0 "
                "(llm.* / img.*) or legacy paligemma_with_expert.* keys."
            )
        print("Sample tensor keys:", names[:6])
    print(
        "Checkpoint header PASSED. This does not validate weight shapes, "
        "runtime memory use, training, or the actual optimizer step."
    )


if __name__ == "__main__":
    main()
