# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Read-only FR3 -> OpenPI pi0.5 input transform preflight.

This tool never starts a controller, loads a policy checkpoint, publishes
robot commands or turns model output into an executable action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.realworld_check.fr3_policy_contract import candidate_19d_state


def prepare_observation(
    measured_state: dict[str, Any],
    base_bgr: np.ndarray,
    wrist_bgr: np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    """Build the *candidate* two-view, 19D inference input from real observations."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Nonempty task prompt required")
    images = {}
    for name, raw in (("observation/image", base_bgr),
                      ("observation/extra_view_image", wrist_bgr)):
        frame = np.asarray(raw)
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"{name} requires HWC uint8 BGR image")
        if min(frame.shape[:2]) < 1:
            raise ValueError(f"{name} cannot be empty")
        images[name] = np.ascontiguousarray(frame[..., ::-1])
    state = candidate_19d_state(measured_state)
    if state.shape != (19,) or state.dtype != np.float32:
        raise ValueError("Invalid candidate FR3 state")
    return {**images, "observation/state": state, "prompt": prompt}


def load_recorded_observation(root: Path, index: int, prompt: str) -> dict[str, Any]:
    """Read one already recorded raw frame; no ROS or camera access."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "raw_recorded" or manifest.get("policy_ready") is not False:
        raise ValueError("A complete, unlabelled raw capture is required")
    if type(index) is not int or index < 0 or index >= manifest.get("frames_saved", 0):
        raise ValueError("Frame index outside saved raw episode")
    with (root / "states.jsonl").open(encoding="utf-8") as src:
        for idx, line in enumerate(src):
            if idx == index:
                record = json.loads(line)
                break
        else:
            raise ValueError("Raw states file shorter than manifest")
    if record.get("frame_index") != index or record.get("training_action") is not None:
        raise ValueError("Unexpected or pre-labelled raw observation")
    paths = record.get("image_files_bgr8", {})
    frames = {}
    for role in ("base", "wrist"):
        expected = f"frames/{role}/{index:06d}.npy"
        if paths.get(role) != expected:
            raise ValueError(f"Unexpected {role} frame path")
        frames[role] = np.load(root / expected, allow_pickle=False)
    return prepare_observation(
        record["measured_state"], frames["base"], frames["wrist"], prompt
    )


def audit_prediction_chunk(chunk: Any) -> dict[str, Any]:
    """Inspect hypothetical decoded 7D model output; NEVER authorize execution."""
    try:
        arr = np.asarray(chunk, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prediction must be numeric") from exc
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 7:
        raise ValueError("Expected a decoded (horizon, 7) action chunk")
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite model output")
    return {
        "decoded_action_shape": list(arr.shape),
        "finite": True,
        "execution_approved": False,
        "reason": "FR3 frame/units/limits and gripper mapping have not been safety qualified",
    }


def validate_openpi_pi05_transform(observation: dict[str, Any]) -> dict[str, Any]:
    """Invoke the *real* existing OpenPI input adapter, not a mock."""
    try:
        from openpi.models.model import ModelType
        from rlinf.models.embodiment.openpi.policies.realworld_policy import RealworldInputs
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "OpenPI runtime missing; select the OpenPI Python environment to verify "
            "the real RealworldInputs transform"
        ) from exc

    transformed = RealworldInputs(
        action_dim=32, model_type=ModelType.PI05
    )(dict(observation))

    state = np.asarray(transformed["state"])
    if state.shape != (32,) or not np.isfinite(state).all():
        raise ValueError("OpenPI state padding failed")
    image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    if set(transformed["image"]) != set(image_keys):
        raise ValueError("Unexpected OpenPI image roles")
    mask = transformed["image_mask"]
    if set(mask) != set(image_keys) or any(
        bool(mask[k]) != expected
        for k, expected in zip(image_keys, (True, True, False))
    ):
        raise ValueError("OpenPI 2-view masks differ from expected config")
    if not np.array_equal(transformed["image"]["base_0_rgb"],
                          observation["observation/image"]):
        raise ValueError("OpenPI base image changed unexpectedly")
    if not np.array_equal(transformed["image"]["left_wrist_0_rgb"],
                          observation["observation/extra_view_image"]):
        raise ValueError("OpenPI wrist image changed unexpectedly")
    return {
        "passed": True,
        "registered_config": "pi05_franka_pnp",
        "transform": "RealworldInputs(ModelType.PI05, action_dim=32)",
        "state_dim_original": 19,
        "state_dim_padded": 32,
        "image_keys": list(image_keys),
        "image_masks": {k: bool(mask[k]) for k in image_keys},
        "real_checkpoint_loaded": False,
        "model_inference_executed": False,
        "robot_commands_sent": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--prompt", default="pick up the object")
    args = parser.parse_args()
    obs = load_recorded_observation(args.raw, args.frame, args.prompt)
    report = validate_openpi_pi05_transform(obs)
    report.update({
        "source_raw_episode": str(args.raw),
        "source_frame": args.frame,
        "candidate_state_not_calibrated": True,
        "control_adapter_ready": False,
        "milestone_5_status": "in_progress_offline_input_transform_verified",
    })
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
