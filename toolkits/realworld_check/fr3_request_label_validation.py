# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Read-only FR3 request-label reconstruction and lag/axis diagnostics.

Validates the existing candidate against the original operator events when
--verify-source is supplied, without changing or recreating LeRobot datasets.
The per-axis motion correlations are descriptive, NOT controller calibration.
Does not import ROS, contact the robot, train a model, or send commands.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from toolkits.realworld_check.fr3_command_request_alignment import (
    SCHEMA, FIELDS, align_episode,
)

AXES = ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av, aw = a[:3], float(a[3])
    bv, bw = b[:3], float(b[3])
    return np.r_[aw * bv + bw * av + np.cross(av, bv), aw * bw - np.dot(av, bv)]


def _relative_world_rotvec(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Rotation mapping q0 to q1, expressed in the state pose world frame."""
    delta = _quat_mul(q1, np.r_[-q0[:3], q0[3]])
    if delta[3] < 0:
        delta = -delta
    v = delta[:3]
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return 2 * v
    return (2 * math.atan2(norm, float(delta[3])) / norm) * v


def _corr(a: list[float], b: list[float]) -> float | None:
    if len(a) < 30:
        return None
    u, v = np.asarray(a), np.asarray(b)
    if float(np.std(u)) < 1e-7 or float(np.std(v)) < 1e-9:
        return None
    result = float(np.corrcoef(u, v)[0, 1])
    return round(result, 4) if math.isfinite(result) else None


def _assert_same_reconstruction(actual: list[dict], expected: list[dict],
                                episode: str) -> None:
    if len(actual) != len(expected):
        raise ValueError(f"{episode}: reconstructed row count differs")
    for i, (old, new) in enumerate(zip(actual, expected, strict=True)):
        for field in ("frame_index", "observation_camera_receipt_ns",
                      "next_observation_camera_receipt_ns", "candidate_complete",
                      "gripper_request_open_fraction"):
            if old.get(field) != new.get(field):
                raise ValueError(f"{episode}:{i}: mismatch in {field}")
        for field in ("state19_candidate", "spacemouse_mean_request_6d",
                      "candidate_request_action_7d"):
            lhs, rhs = old.get(field), new.get(field)
            if lhs is None or rhs is None:
                if lhs is not rhs:
                    raise ValueError(f"{episode}:{i}: mismatch in {field}")
            elif not np.allclose(lhs, rhs, atol=1e-6, rtol=1e-6):
                raise ValueError(f"{episode}:{i}: mismatch in {field}")


def validate(candidate_root: Path, *, verify_source: bool = False,
             max_lag_frames: int = 5) -> dict:
    if not 0 <= max_lag_frames <= 10:
        raise ValueError("max_lag_frames must be 0..10")
    root = candidate_root.expanduser().resolve()
    manifest = _load_json(root / "report.json")
    if manifest.get("schema") != SCHEMA or manifest.get("action_fields") != list(FIELDS):
        raise ValueError("Not the expected original 7D normalized-request dataset")
    raw = Path(manifest["source_root"]).expanduser().resolve()
    episodes = manifest.get("episode_reports", [])
    if not episodes or len(episodes) != 100:
        raise ValueError("Expected the 100-demonstration source report")

    pairs = defaultdict(lambda: {"request": [], "measured": []})
    totals = {"episodes": 0, "rows": 0, "complete_labels": 0,
              "episode_adjacencies": 0, "reconstructed_episodes": 0}
    dt = []
    for ep in episodes:
        name = ep["episode"]
        if not isinstance(name, str) or not name.startswith("episode_") or "/" in name or ".." in name:
            raise ValueError("Invalid source episode name")
        path = root / "episodes" / (name + ".jsonl")
        rows = _read_rows(path)
        if len(rows) != ep["transitions"]:
            raise ValueError(f"{name}: length changed from report")
        if verify_source:
            rebuilt_report, reconstructed = align_episode(
                raw / name,
                max_age_s=float(manifest["max_request_age_s"]),
                min_coverage=float(manifest["min_twist_time_coverage"]),
            )
            if (rebuilt_report["recorded_requests_sha256"] != ep["recorded_requests_sha256"]
                    or rebuilt_report["raw_manifest_sha256"] != ep["raw_manifest_sha256"]):
                raise ValueError(f"{name}: raw source changed")
            _assert_same_reconstruction(rows, reconstructed, name)
            totals["reconstructed_episodes"] += 1

        commands: list[np.ndarray | None] = []
        positions = []
        quats = []
        for i, row in enumerate(rows):
            start, end = (row.get("observation_camera_receipt_ns"),
                          row.get("next_observation_camera_receipt_ns"))
            if (row.get("frame_index") != i or row.get("next_frame_index") != i + 1
                    or type(start) is not int or type(end) is not int
                    or start >= end or not 0.02 <= (end - start) / 1e9 <= 0.25):
                raise ValueError(f"{name}:{i}: invalid frame timing")
            if i and rows[i - 1]["next_observation_camera_receipt_ns"] != start:
                raise ValueError(f"{name}:{i}: missing/noncontiguous observation")
            d = (end - start) / 1e9
            dt.append(d)
            state = np.asarray(row.get("state19_candidate"), dtype=float)
            if state.shape != (19,) or not np.isfinite(state).all():
                raise ValueError(f"{name}:{i}: invalid state")
            if abs(np.linalg.norm(state[3:7]) - 1) > 0.05:
                raise ValueError(f"{name}:{i}: invalid TCP quaternion")
            act = row.get("candidate_request_action_7d")
            if row.get("candidate_complete") is True:
                a = np.asarray(act, dtype=float)
                if (a.shape != (7,) or not np.isfinite(a).all()
                        or np.max(np.abs(a[:6])) > 1.001 or a[6] not in (0., 1.)):
                    raise ValueError(f"{name}:{i}: invalid normalized request")
                if (not np.allclose(a[:6], row.get("spacemouse_mean_request_6d"))
                        or a[6] != row.get("gripper_request_open_fraction")):
                    raise ValueError(f"{name}:{i}: label does not match row fields")
                commands.append(a[:6])
                totals["complete_labels"] += 1
            elif act is None:
                commands.append(None)
            else:
                raise ValueError(f"{name}:{i}: incomplete label was not null")
            positions.append(state[:3])
            quats.append(state[3:7])
        velocities = []
        for i in range(len(rows) - 1):
            d = (rows[i]["next_observation_camera_receipt_ns"]
                 - rows[i]["observation_camera_receipt_ns"]) / 1e9
            v = (positions[i + 1] - positions[i]) / d
            w = _relative_world_rotvec(quats[i], quats[i + 1]) / d
            velocities.append(np.r_[v, w])
            totals["episode_adjacencies"] += 1
        for i, action in enumerate(commands):
            if action is None:
                continue
            for lag in range(max_lag_frames + 1):
                if i + lag >= len(velocities):
                    break
                motion = velocities[i + lag]
                for axis in range(6):
                    entry = pairs[(axis, lag)]
                    entry["request"].append(float(action[axis]))
                    entry["measured"].append(float(motion[axis]))
        totals["episodes"] += 1
        totals["rows"] += len(rows)

    if (totals["complete_labels"] != manifest.get("candidate_7d_available")
            or totals["rows"] != manifest.get("transitions")):
        raise ValueError("Counts do not match original alignment report")
    associations = []
    for axis, name in enumerate(AXES):
        lag_results = []
        for lag in range(max_lag_frames + 1):
            sample = pairs[(axis, lag)]
            correlation = _corr(sample["request"], sample["measured"])
            lag_results.append({
                "lag_frames": lag, "sample_pairs": len(sample["request"]),
                "pearson_r": correlation,
            })
        usable = [x for x in lag_results if x["pearson_r"] is not None]
        best = max(usable, key=lambda x: abs(x["pearson_r"])) if usable else None
        associations.append({
            "channel": name,
            "measured_component": (
                "world_tcp_translation" if axis < 3 else "world_tcp_rotation"
            ),
            "best_absolute_correlation": best,
            "all_lags": lag_results,
        })
    return {
        "schema": "fr3_request_label_offline_validation_v1",
        "candidate_root": str(root),
        "source_event_reconstruction_passed": verify_source,
        "dataset_integrity_passed": True,
        **totals,
        "median_observation_interval_s": round(float(np.median(dt)), 5),
        "association_only": associations,
        "label_timing": (
            "6D operator requests averaged over [camera receipt t_i, t_i+1]; "
            "7th channel is latched gripper request at t_i+1; "
            "state and images correspond to observation receipt t_i"
        ),
        "caveats": [
            "Matching axes are not proof of frame calibration: mounting rotation, camera lag, controller hold/IK and co-active axes may distort correlations.",
            "Observed displacement is NOT controller-accepted action or command target.",
            "Camera receipt time is not calibrated exposure time; workstation receipt is not NUC command acceptance time.",
            "Documented angular scaling disagrees across historical inspections (0.03 vs 0.06); inspect running NUC binary/source and configured parameters.",
        ],
        "controller_frame_scale_verified": False,
        "controller_acceptance_verified": False,
        "exposure_and_cross_host_latency_verified": False,
        "physical_execution_approved": False,
        "full_3000_step_sft_gate_passed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--verify-source", action="store_true",
                        help="Reconstruct every row from immutable raw source events (read-only)")
    parser.add_argument("--max-lag-frames", type=int, default=5)
    args = parser.parse_args()
    report = validate(args.candidate_root, verify_source=args.verify_source,
                      max_lag_frames=args.max_lag_frames)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
