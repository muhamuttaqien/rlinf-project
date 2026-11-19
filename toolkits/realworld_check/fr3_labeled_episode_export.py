# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline FR3 LeRobot exporter for FUTURE reviewed, controller-accepted labels.

Requires a separate per-frame labels sidecar bound to an immutable raw capture
manifest. The stationary raw pilots have no labels and cannot be exported.
This module has NO ROS/hardware/policy-controller interface and never invents
actions from SpaceMouse intents, observed pose changes or a Panda model.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.realworld_check.fr3_lerobot_preflight import audit_raw_episode
from toolkits.realworld_check.fr3_policy_contract import (
    ACTION_FIELDS, CONTRACT_VERSION, candidate_19d_state,
)

SCHEMA = "fr3_controller_accepted_training_labels_v1"
ACTION_SOURCE = "controller_accepted_metric_tcp_delta_and_robotiq_knuckle_target"
REVIEW_KEYS = (
    "tcp_base_frame_and_tool_offset_confirmed",
    "rotation_vector_convention_confirmed",
    "robotiq_knuckle_target_units_confirmed",
    "controller_accepted_action_source_confirmed",
    "cross_host_clock_calibration_confirmed",
)


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(field + " must be an integer with valid range")
    return value


def _load_json_object(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object at " + str(path))
    return value


def check_labels(raw: Path, labels_path: Path) -> tuple[dict, list[dict]]:
    """Validate provenance, action semantics, 1:1 alignment and human review.

    Boolean attestations are operator assertions, not independent evidence of
    physical controller correctness or hardware safety. No output is written.
    """
    raw = Path(raw)
    prior, _ = audit_raw_episode(raw)
    manifest_path = raw / "manifest.json"
    manifest = _load_json_object(manifest_path)
    labels = _load_json_object(labels_path)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    if labels.get("schema") != SCHEMA:
        raise ValueError("Unsupported labels schema")
    if labels.get("raw_manifest_sha256") != digest:
        raise ValueError("Labels do not match raw manifest SHA256")
    if labels.get("task") != manifest.get("task"):
        raise ValueError("Task differs from captured episode")
    if labels.get("state_contract") != CONTRACT_VERSION:
        raise ValueError("Wrong FR3 candidate state contract")
    if labels.get("action_field_names") != list(ACTION_FIELDS):
        raise ValueError("Action channel semantics mismatch")
    if labels.get("action_source") != ACTION_SOURCE:
        raise ValueError("Normalized SpaceMouse intent or inferred pose is not an action label")
    if type(labels.get("episode_success")) is not bool:
        raise ValueError("Human-reviewed episode_success boolean required")
    if labels.get("human_reviewed") is not True or not isinstance(
        labels.get("reviewer"), str
    ) or not labels["reviewer"].strip():
        raise ValueError("Explicit human outcome/label reviewer required")
    source = labels.get("controller_telemetry_source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Traceable external controller-telemetry source required")
    verification = labels.get("verification")
    if not isinstance(verification, dict) or set(verification) != set(REVIEW_KEYS):
        raise ValueError("Missing/unknown action verification checklist")
    if any(verification[key] is not True for key in REVIEW_KEYS):
        raise ValueError("Controller, frame and timebase verification must be attested")
    clock = labels.get("clock")
    if not isinstance(clock, dict) or set(clock) != {
        "nuc_minus_workstation_monotonic_ns",
        "uncertainty_ns", "max_skew_ns"
    }:
        raise ValueError("Explicit calibrated clock metadata required")
    offset = clock["nuc_minus_workstation_monotonic_ns"]
    if type(offset) is not int:
        raise ValueError("Measured NUC-to-workstation clock offset required")
    uncertainty = _positive_int(clock["uncertainty_ns"], "uncertainty_ns", allow_zero=True)
    tolerance = _positive_int(clock["max_skew_ns"], "max_skew_ns")
    if uncertainty >= tolerance:
        raise ValueError("Clock uncertainty exceeds alignment tolerance")

    rows = labels.get("records")
    if not isinstance(rows, list) or len(rows) != prior["sample_count"]:
        raise ValueError("One accepted target record required for EVERY observation")
    raw_records = [
        json.loads(line) for line in (raw / "states.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    stamps: list[int] = []
    target_ids = set()
    checked: list[dict] = []
    for index, (row, observation) in enumerate(zip(rows, raw_records, strict=True)):
        if not isinstance(row, dict) or set(row) != {
            "frame_index", "controller_accepted_ns", "source_record_id", "action"
        }:
            raise ValueError("Unexpected action-label fields")
        if row["frame_index"] != index or type(row["frame_index"]) is not int:
            raise ValueError("Action frame order/count mismatch")
        stamp = _positive_int(row["controller_accepted_ns"], "controller_accepted_ns")
        observed = _positive_int(
            observation["recorded_monotonic_ns"], "recorded_monotonic_ns"
        )
        if abs(stamp - (observed + offset)) + uncertainty > tolerance:
            raise ValueError("Controller-to-observation clock skew exceeds tolerance")
        if stamps and stamp <= stamps[-1]:
            raise ValueError("Accepted controller timestamps must be increasing")
        stamps.append(stamp)
        ident = row["source_record_id"]
        if not isinstance(ident, str) or not ident.strip() or ident in target_ids:
            raise ValueError("Target source record IDs must be nonempty and unique")
        target_ids.add(ident)
        try:
            action = np.asarray(row["action"], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Non-numeric action") from exc
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Seven finite physical action values required")
        checked.append({
            "index": index,
            "action": action.astype(np.float32),
            "state": candidate_19d_state(observation["measured_state"]),
            "source_record_id": ident,
        })

    report = {
        "labels_schema": SCHEMA,
        "raw_capture_validated": True,
        "frames": len(checked),
        "state_shape": [len(checked), 19],
        "action_shape": [len(checked), 7],
        "clock_alignment": "explicit measured-offset sidecar, checked for all frames",
        "frame_and_unit_attestations": "operator-provided; not independently hardware-verified",
        "episode_success_human_reviewed": labels["episode_success"],
        "label_source": ACTION_SOURCE,
        "policy_training_data_exportable": True,
        "physical_policy_execution_approved": False,
        "robot_commands_sent": False,
    }
    return report, checked


def export_lerobot(raw: Path, labels_path: Path, output: Path, repo_id: str) -> dict:
    """Write one new LeRobot episode only after ALL sidecar checks pass."""
    report, checked = check_labels(raw, labels_path)
    if not isinstance(repo_id, str) or not repo_id.strip() or "/" not in repo_id:
        raise ValueError("Explicit local/owner repo_id required")
    raw = Path(raw)
    manifest = _load_json_object(raw / "manifest.json")
    labels = _load_json_object(labels_path)
    profiles = manifest["camera_profiles"]

    features = {
        "image": {
            "dtype": "image",
            "shape": (profiles["base"]["resolution"][1],
                      profiles["base"]["resolution"][0], 3),
            "names": ["height", "width", "channel"],
        },
        "extra_view_image": {
            "dtype": "image",
            "shape": (profiles["wrist"]["resolution"][1],
                      profiles["wrist"]["resolution"][0], 3),
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (19,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
        "is_success": {"dtype": "bool", "shape": (1,), "names": ["is_success"]},
    }

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from rlinf.data.storage.lerobot.compat import add_frame_to_dataset

    output = Path(output).expanduser()
    # The output is never reused or overwritten.
    if output.exists():
        raise FileExistsError(output)
    kwargs = {
        "repo_id": repo_id, "root": output, "fps": manifest["sample_rate_hz_requested"],
        "robot_type": "fr3_robotiq", "features": features,
        "image_writer_threads": 1, "image_writer_processes": 0,
    }
    if "use_videos" in inspect.signature(LeRobotDataset.create).parameters:
        kwargs["use_videos"] = False
    dataset = LeRobotDataset.create(**kwargs)
    try:
        for index, row in enumerate(checked):
            images = {}
            for role, target_key in (("base", "image"), ("wrist", "extra_view_image")):
                source = raw / "frames" / role / (f"{index:06d}.npy")
                bgr = np.load(source, allow_pickle=False)
                images[target_key] = np.ascontiguousarray(bgr[..., ::-1])
            terminal = index == len(checked) - 1
            add_frame_to_dataset(dataset, {
                **images,
                "state": row["state"], "actions": row["action"],
                "done": np.array([terminal], dtype=np.bool_),
                "is_success": np.array([
                    terminal and labels["episode_success"]
                ], dtype=np.bool_),
                "task": manifest["task"],
            })
        dataset.save_episode()
        if getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.stop()
            dataset.image_writer = None
        del dataset

    provenance = {
        **report,
        "raw_manifest_sha256": labels["raw_manifest_sha256"],
        "label_sidecar_sha256": hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
        "repo_id": repo_id,
        "source_trace_ids": [r["source_record_id"] for r in checked],
        "note": "Requires human/engineering verification; NOT authority to command a robot",
    }
    (output / "fr3_label_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True,
                        help="FUTURE external controller-accepted action sidecar")
    parser.add_argument("--output", type=Path, required=True,
                        help="NEW LeRobot directory (no overwrite)")
    parser.add_argument("--repo-id", required=True,
                        help="LeRobot repo-id, e.g. local/fr3_recorded")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate sidecar without writing a dataset")
    args = parser.parse_args()
    if args.validate_only:
        report, _ = check_labels(args.raw, args.labels)
    else:
        report = export_lerobot(args.raw, args.labels, args.output, args.repo_id)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
