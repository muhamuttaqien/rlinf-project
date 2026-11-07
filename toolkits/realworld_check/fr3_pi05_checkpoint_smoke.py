# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""OFFLINE computational pi0.5 checkpoint smoke against recorded FR3 frames.

Loads a user-provided *simulation-trained* checkpoint and evaluates the
network once using actual recorded two-camera/robot state and ONE fixed valid
placeholder language token. It does NOT test real text conditioning, model
normalization, calibration, controller safety or real robot actions.
No ROS imports, no policy server, no robot connections or publishers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from toolkits.realworld_check.fr3_pi05_inference_preflight import (
    audit_prediction_chunk, load_recorded_observation,
)


def resolve_checkpoint(path: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_file() and p.name == "full_weights.pt":
        return p
    for rel in ("actor/model_state_dict/full_weights.pt",
                "model_state_dict/full_weights.pt", "full_weights.pt"):
        candidate = p / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No RLinf full_weights.pt under {p}")


def run_once(raw: Path, checkpoint: Path, *, num_steps: int = 2,
             horizon: int = 10, device: str = "cuda:0") -> dict:
    if num_steps < 1 or num_steps > 10:
        raise ValueError("Set num_steps between 1 and 10 for an offline smoke")
    if horizon < 1 or horizon > 50:
        raise ValueError("Invalid checkpoint action horizon")
    checkpoint = resolve_checkpoint(checkpoint)

    import torch

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("This multi-billion-parameter smoke requires CUDA; no CPU fallback")
    from openpi.models.model import ModelType

    from rlinf.models.embodiment.openpi.checkpoint import load_full_weights
    from rlinf.models.embodiment.openpi.modules.model import Observation
    from rlinf.models.embodiment.openpi.pi0 import Pi0
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi.policies.realworld_policy import RealworldInputs

    raw_obs = load_recorded_observation(raw, 0, "SYNTHETIC TOKEN - offline smoke")
    converted = RealworldInputs(action_dim=32, model_type=ModelType.PI05)(raw_obs)

    def to_image(value):
        arr = np.asarray(value)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("Expected raw RGB uint8 HWC")
        return (torch.from_numpy(np.ascontiguousarray(arr))
                .to(device=device, dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0)

    # Critical: fixed token is a syntactically valid embedding index, NOT the
    # prompt encoding. This only proves checkpoint+vision+state forward.
    observation = Observation(
        images={name: to_image(arr) for name, arr in converted["image"].items()},
        image_masks={name: torch.tensor([bool(mask)], dtype=torch.bool, device=device)
                     for name, mask in converted["image_mask"].items()},
        state=torch.as_tensor(np.asarray(converted["state"]),
                              dtype=torch.float32, device=device).reshape(1, 32),
        tokenized_prompt=torch.ones((1, 1), dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.ones((1, 1), dtype=torch.bool, device=device),
    )

    # Model dimensions mirror the existing StackCube pi0.5 setup, but they are
    # NOT inferred from the checkpoint (which is a tensor-only state dict).
    # The loader rejects mismatched backbone tensor shapes.
    config = Pi0Config(
        pi05=True, action_dim=32, action_horizon=horizon,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        discrete_state_input=False,
        dtype="bfloat16",
    )
    model = Pi0(config)
    load_full_weights(model, checkpoint, expect_rlt=False)
    model.to_bfloat16_for_selected_params("bfloat16")
    model.to(device)
    model.eval()

    with torch.inference_mode():
        predicted = model.sample_actions(observation, num_steps=num_steps)
    shape = list(predicted.shape)
    if shape != [1, horizon, 32]:
        raise ValueError(f"Unexpected raw π0.5 output tensor shape {shape}")
    output = predicted.detach().float().cpu().numpy()[0, :, :7]
    audit = audit_prediction_chunk(output)

    return {
        "status": "checkpoint_math_smoke_passed",
        "checkpoint": str(checkpoint),
        "checkpoint_task": "StackCube simulation, NOT FR3",
        "source_raw": str(raw),
        "input_two_real_camera_frames": True,
        "input_candidate_19d_state": True,
        "input_text_tokenization_verified": False,
        "language_token": "synthetic placeholder, not prompt tokenization",
        "policy_normalization_verified": False,
        "model_weight_load_success": True,
        "pi05_forward_executed": True,
        "raw_output_shape": shape,
        "decoded_first_seven": audit,
        "fr3_action_semantics_verified": False,
        "robot_commands_sent": False,
        "physical_execution_approved": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True,
                        help="Existing completed raw FR3 recording")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Local RLinf StackCube full_weights.pt or step folder")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=10,
                        help="Network action horizon; default StackCube candidate 10")
    args = parser.parse_args()
    result = run_once(
        args.raw, args.checkpoint, device=args.device,
        num_steps=args.num_steps, horizon=args.horizon,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
