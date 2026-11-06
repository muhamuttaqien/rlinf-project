# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline FR3 data-path smoke test. Never connects to ROS or commands hardware.

Distinguish:
  (a) synthetic, calibrated-clock alignment + local LeRobot writer roundtrip;
  (b) existing real raw-observation validation;
  (c) NOT YET VERIFIED real cross-host synchronization or executed 7D actions.

All synthetic actions are marked as synthetic and never written to a real
episode. LeRobot output is temporary and removed when this command exits.
"""

from __future__ import annotations

import argparse
import bisect
import inspect
import json
import tempfile
from pathlib import Path

import numpy as np

from toolkits.realworld_check.fr3_lerobot_preflight import audit_raw_episode
from toolkits.realworld_check.franka_pi05_contract import validate_realworld_pi05_sample


def align_by_calibrated_clock(
    observation_monotonic_ns: list[int],
    controller_tick_ns: list[int],
    *,
    nuc_minus_workstation_ns: int,
    uncertainty_ns: int,
    max_skew_ns: int = 20_000_000,
) -> list[tuple[int, int]]:
    """Associate each observation with nearest controller tick.

    NUC tick time = workstation receipt time + *measured* clock offset.
    Offset MUST originate from a real clock-calibration protocol; never
    estimate it from separate, non-overlapping recordings. Returned pairs
    are (controller tick index, signed residual in nanoseconds).
    """
    if not isinstance(nuc_minus_workstation_ns, int):
        raise ValueError("A measured integer clock offset is required")
    if (type(uncertainty_ns) is not int or uncertainty_ns < 0
            or uncertainty_ns >= max_skew_ns or max_skew_ns <= 0):
        raise ValueError("Clock uncertainty must be smaller than skew tolerance")

    def check(stamps: list[int], label: str) -> None:
        if not stamps or any(type(v) is not int or v <= 0 for v in stamps):
            raise ValueError(f"{label} must contain positive integer timestamps")
        if any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"{label} timestamps must be strictly increasing")

    check(observation_monotonic_ns, "observation")
    check(controller_tick_ns, "controller")
    aligned = []
    for stamp in observation_monotonic_ns:
        desired = stamp + nuc_minus_workstation_ns
        i = bisect.bisect_left(controller_tick_ns, desired)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(controller_tick_ns)]
        best = min(candidates, key=lambda j: abs(controller_tick_ns[j] - desired))
        residual = controller_tick_ns[best] - desired
        if abs(residual) + uncertainty_ns > max_skew_ns:
            raise ValueError("Clock-aligned observation exceeds skew tolerance")
        aligned.append((best, residual))
    if any(b[0] <= a[0] for a, b in zip(aligned, aligned[1:])):
        raise ValueError("Multiple observation frames mapped to one/out-of-order tick")
    return aligned


def synthetic_alignment_smoke() -> dict:
    # Two deliberately different monotonic clocks, with a known artificial
    # 5-second offset. No real clock calibration is claimed.
    workstation = [10_000_000_000 + i * 100_000_000 for i in range(3)]
    nuc_ticks = [v + 5_000_000_000 + 1_000_000 for v in workstation]
    aligned = align_by_calibrated_clock(
        workstation, nuc_ticks,
        nuc_minus_workstation_ns=5_000_000_000,
        uncertainty_ns=500_000,
        max_skew_ns=5_000_000,
    )
    assert [idx for idx, _ in aligned] == [0, 1, 2]
    return {"passed": True, "frames": 3, "max_synthetic_residual_ns": 1_000_000}


def _rgb_image(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise ValueError("Decoded RGB image has invalid normalized range")
        arr = np.rint(arr * 255).astype(np.uint8)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("LeRobot returned invalid decoded RGB image")
    return arr


def lerobot_writer_roundtrip() -> dict:
    """Actually write and reopen one temporary synthetic LeRobot episode."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.data.storage.lerobot.compat import add_frame_to_dataset

    features = {
        "image": {"dtype": "image", "shape": (64, 64, 3),
                  "names": ["height", "width", "channel"]},
        "extra_view_image": {"dtype": "image", "shape": (64, 64, 3),
                             "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (19,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
        "is_success": {"dtype": "bool", "shape": (1,), "names": ["is_success"]},
    }

    with tempfile.TemporaryDirectory(prefix="fr3_setup_synthetic_") as tmp:
        root = Path(tmp) / "dataset"
        kwargs = dict(repo_id="local/fr3_setup_synthetic", root=root,
                      fps=10, robot_type="fr3_robotiq", features=features,
                      image_writer_threads=1, image_writer_processes=0)
        if "use_videos" in inspect.signature(LeRobotDataset.create).parameters:
            kwargs["use_videos"] = False
        ds = LeRobotDataset.create(**kwargs)
        for i in range(3):
            state = np.zeros(19, dtype=np.float32)
            state[6] = 1  # xyzw identity quaternion (synthetic)
            frame = {
                "image": np.full((64, 64, 3), i * 10, dtype=np.uint8),
                "extra_view_image": np.full((64, 64, 3), 50 + i * 10, dtype=np.uint8),
                "state": state,
                "actions": np.zeros(7, dtype=np.float32),
                "done": np.array([i == 2], dtype=np.bool_),
                "is_success": np.array([False], dtype=np.bool_),
                "task": "SYNTHETIC ONLY: FR3 dataset setup smoke",
            }
            add_frame_to_dataset(ds, frame)
        ds.save_episode()
        if getattr(ds, "image_writer", None) is not None:
            ds.image_writer.wait_until_done()
        # Close writer resources before re-opening.
        if getattr(ds, "image_writer", None) is not None:
            ds.image_writer.stop()
            ds.image_writer = None
        del ds

        read = LeRobotDataset("local/fr3_setup_synthetic",
                              root=root, download_videos=False)
        if len(read) != 3:
            raise AssertionError(f"LeRobot reload returned {len(read)} frames, expected 3")
        row = read[0]
        state = np.asarray(row["state"], dtype=np.float32)
        action = np.asarray(row["actions"], dtype=np.float32)
        sample = {
            "observation/image": _rgb_image(row["image"]),
            "observation/extra_view_image": _rgb_image(row["extra_view_image"]),
            "observation/state": state,
            "actions": action.reshape(1, 7),
            "prompt": "SYNTHETIC ONLY: FR3 dataset setup smoke",
        }
        validate_realworld_pi05_sample(sample)

    return {"passed": True, "synthetic_episode_frames": 3,
            "openpi_input_shapes_passed": True,
            "warning": "Synthetic only; not a verified full OpenPI dataloader or RECAP run"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path,
                   help="Optional existing raw FR3 capture to validate (read-only)")
    args = p.parse_args()

    result = {
        "stage": "offline_setup_smoke",
        "synthetic_clock_alignment": synthetic_alignment_smoke(),
        "synthetic_lerobot_roundtrip": None,
        "raw_episode_preflight": None,
        "real_cross_host_alignment_verified": False,
        "real_training_actions_available": False,
        "recap_two_camera_loader_verified": False,
        "real_demonstration_performed": False,
    }
    if args.raw:
        report, _ = audit_raw_episode(args.raw)
        result["raw_episode_preflight"] = {
            "passed": bool(report["raw_capture_valid"]),
            "frames": report["sample_count"],
            "operator_events": report["operator_event_count"],
        }

    try:
        result["synthetic_lerobot_roundtrip"] = lerobot_writer_roundtrip()
    except Exception as exc:
        result["synthetic_lerobot_roundtrip"] = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(result, indent=2))
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
