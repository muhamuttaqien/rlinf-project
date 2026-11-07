# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""OFFLINE computational pi0.5 checkpoint smoke against recorded FR3 frames.

Loads a user-provided *simulation-trained* checkpoint and evaluates the
network once using recorded two-camera/robot state and official locally cached
PaliGemma tokenization. Simulation normalization is audited but NEVER applied
to FR3. Does not validate robot safety, real-world policy or actions.
No ROS imports, no policy server, no robot connections or publishers.
"""

from __future__ import annotations

import argparse
import json
import os
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


def official_tokenizer_cache_path() -> Path:
    """Expected official OpenPI PaliGemma tokenizer cache, no network."""
    home = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    return home / "big_vision" / "paligemma_tokenizer.model"


def tokenize_prompt_offline(prompt: str) -> tuple[np.ndarray, np.ndarray]:
    cache = official_tokenizer_cache_path()
    if not cache.is_file():
        raise FileNotFoundError(
            f"Missing OpenPI PaliGemma tokenizer at {cache}. "
            "Do not substitute Gemma-3-270m or SigLIP2 tokenizers. "
            "No implicit download attempted."
        )
    from openpi.models.tokenizer import PaligemmaTokenizer
    ids, mask = PaligemmaTokenizer(max_len=200).tokenize(prompt, state=None)
    ids = np.asarray(ids, dtype=np.int64)
    mask = np.asarray(mask, dtype=np.bool_)
    if ids.shape != (200,) or mask.shape != (200,):
        raise ValueError("Unexpected OpenPI tokenizer shapes")
    n = int(mask.sum())
    if n < 1 or n >= 200 or not np.array_equal(
        mask, np.arange(200) < n
    ):
        raise ValueError("Invalid or truncated PaliGemma token sequence")
    return ids[:n].copy(), mask[:n].copy()


def audit_sim_norm_stats(path: Path) -> dict:
    """Inspect dimensionality; simulation stats are NEVER applied to FR3."""
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    raw = raw.get("norm_stats", raw)
    if not isinstance(raw, dict):
        raise ValueError("Expected norm stats object")
    dims = {}
    for key in ("state", "actions", "action"):
        item = raw.get(key)
        if not isinstance(item, dict):
            continue
        values = item.get("mean", item.get("q01"))
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"Invalid norm stats {key}")
        dims[key] = int(array.size)
    return {
        "observed_dimensions": dims,
        "required_state_dim": 19,
        "required_action_dim": 7,
        "dimension_match": (
            dims.get("state") == 19 and
            dims.get("actions", dims.get("action")) == 7
        ),
        "applied_to_inference": False,
        "approved_for_FR3": False,
        "reason": "StackCube simulation stats do not calibrate real FR3",
    }


def run_once(raw: Path, checkpoint: Path, *, num_steps: int = 2,
             horizon: int = 10, device: str = "cuda:0",
             prompt: str = "pick up the object",
             norm_stats: Path | None = None) -> dict:
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

    raw_obs = load_recorded_observation(raw, 0, prompt)
    tokens, token_masks = tokenize_prompt_offline(prompt)
    converted = RealworldInputs(action_dim=32, model_type=ModelType.PI05)(raw_obs)

    def to_image(value):
        arr = np.asarray(value)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError("Expected raw RGB uint8 HWC")
        return (torch.from_numpy(np.ascontiguousarray(arr))
                .to(device=device, dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0)

    # Official locally cached PaliGemma tokenizer, not a placeholder token.
    observation = Observation(
        images={name: to_image(arr) for name, arr in converted["image"].items()},
        image_masks={name: torch.tensor([bool(mask)], dtype=torch.bool, device=device)
                     for name, mask in converted["image_mask"].items()},
        state=torch.as_tensor(np.asarray(converted["state"]),
                              dtype=torch.float32, device=device).reshape(1, 32),
        tokenized_prompt=torch.as_tensor(tokens[None, :].copy(),
                                         dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.as_tensor(token_masks[None, :].copy(),
                                              dtype=torch.bool, device=device),
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
    from toolkits.realworld_check.fr3_pi05_action_boundary import (
        quarantine_pi05_prediction,
    )
    quarantine = quarantine_pi05_prediction(
        output, checkpoint=str(checkpoint), source_episode=str(raw)
    )

    return {
        "status": "checkpoint_math_smoke_passed",
        "checkpoint": str(checkpoint),
        "checkpoint_task": "StackCube simulation, NOT FR3",
        "source_raw": str(raw),
        "input_two_real_camera_frames": True,
        "input_candidate_19d_state": True,
        "input_text_tokenization_verified": True,
        "language_token": "official cached OpenPI PaliGemma tokenizer",
        "prompt_token_count": int(tokens.size),
        "policy_normalization_verified": False,
        "sim_norm_audit": audit_sim_norm_stats(norm_stats) if norm_stats else None,
        "model_weight_load_success": True,
        "pi05_forward_executed": True,
        "raw_output_shape": shape,
        "decoded_first_seven": audit,
        "output_action_quarantine": quarantine,
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
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--norm-stats", type=Path, default=None,
                        help="Read-only audit; simulator stats are NEVER applied")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=10,
                        help="Network action horizon; default StackCube candidate 10")
    args = parser.parse_args()
    result = run_once(
        args.raw, args.checkpoint, device=args.device,
        num_steps=args.num_steps, horizon=args.horizon,
        prompt=args.prompt, norm_stats=args.norm_stats,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
