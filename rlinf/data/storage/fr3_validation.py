# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only quality assessment of native synthetic FR3 trajectories and journals."""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from rlinf.data.storage.fr3_synthetic import (
    GATE,
    PROVENANCE_SCHEMA,
    SCHEMA,
    _parse_json,
    _validate_action,
    _validate_observation,
    file_digest,
    read_json,
    validate_synthetic_dataset,
)
from rlinf.robotics.parts.fr3_fixtures import SYNTHETIC_EPOCH_NS, validate_alignment
from rlinf.robotics.parts.transports.ros2.action_contract import (
    FR3_KIND,
    ROBOTIQ_KIND,
    ObservationProvenance,
    validate_observation,
)
from rlinf.robotics.parts.transports.ros2.protocol import FR3_JOINTS

QUALITY_SCHEMA = "fr3_dataset_quality_v1"
VALIDATOR_VERSION = "fr3-offline-validator-v1"
# Fixture tolerances only; changing a manifest cannot relax the validator ceiling.
MAX_AGE_NS = 50_000_000
MAX_SKEW_NS = 20_000_000
STATE_SPEC = {
    "arm_joint_effort": (7, "N m"),
    "arm_joint_position": (7, "rad"),
    "arm_joint_velocity": (7, "rad/s"),
    "base_force": (3, "N"),
    "base_torque": (3, "N m"),
    "knuckle_position_rad": (1, "rad"),
    "normalized_closure": (1, "dimensionless [0,1]"),
    "tcp_force": (3, "N"),
    "tcp_pose": (7, "xyz m; quaternion xyzw"),
    "tcp_torque": (3, "N m"),
}


def require(condition: bool, message: str) -> None:
    """Raise a diagnostic when a stored contract is not satisfied."""
    if not condition:
        raise ValueError(message)


def artifact(root: Path, relative: str) -> Path:
    """Resolve a dataset-owned regular file, rejecting traversal and symlinks."""
    require(isinstance(relative, str), "Expected a relative artifact path")
    path = root / relative
    require(
        not Path(relative).is_absolute() and ".." not in Path(relative).parts,
        f"Unsafe artifact path: {relative}",
    )
    require(
        not any(p.is_symlink() for p in (path, *path.parents)),
        f"Symlink artifact: {relative}",
    )
    require(path.is_file(), f"Missing artifact: {relative}")
    return path


def load_records(path: Path) -> list[dict]:
    """Read complete JSONL records without repairing partial writes."""
    lines = path.read_text().splitlines(keepends=True)
    require(all(s.endswith("\n") for s in lines), "Incomplete JSONL record")
    records = [_parse_json(s) for s in lines]
    require(all(isinstance(r, dict) for r in records), "Expected JSONL objects")
    return records


def native_path(root: Path, index: dict, number: int) -> Path:
    """Resolve a trajectory through its native replay index."""
    info = index["trajectory_index"][str(number)]
    return artifact(root, f"demos/trajectory_{number}_{info['model_weights_id']}.pt")


def _load_native(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(isinstance(payload, dict), "Expected trajectory dictionary")
        return payload
    except Exception as exc:
        raise ValueError(f"Corrupted native trajectory {path.name}: {exc}") from exc


def _numeric(value: Any, size: int, field: str) -> np.ndarray:
    values = np.asarray(value)
    require(
        values.shape == (size,)
        and values.dtype.kind in "fiu"
        and np.isfinite(values).all(),
        f"Expected {field}: {size} finite numeric values; found {value!r}",
    )
    return values


def _ns(value: Any) -> int:
    require(
        type(value) is int and value > 0,
        f"Expected positive integer ns; found {value!r}",
    )
    return value


def _gate(record: dict) -> None:
    require(
        all(type(record.get(k)) is type(v) and record[k] == v for k, v in GATE.items()),
        "Expected explicit synthetic / physical_execution=false / NO_GO / NOT_GRANTED gates",
    )


def _tensor_contract(payload: dict, count: int) -> None:
    for key, size, dtype in (
        ("actions", 7, torch.float32),
        ("rewards", 1, torch.float32),
        ("intervene_flags", 7, torch.bool),
        ("dones", 1, torch.bool),
        ("terminations", 1, torch.bool),
        ("truncations", 1, torch.bool),
    ):
        t = payload[key]
        require(
            t.shape == (count, 1, size) and t.dtype == dtype,
            f"Expected {key} [{count},1,{size}] {dtype}; found {t.shape} {t.dtype}",
        )
        require(torch.isfinite(t).all().item(), f"Nonfinite tensor: {key}")
    require(
        torch.equal(payload["forward_inputs"]["action"], payload["actions"]),
        "forward_inputs.action differs from submitted actions",
    )
    require(
        payload["forward_inputs"]["action"].dtype == torch.float32,
        "Wrong forward action dtype",
    )
    require(
        not payload["intervene_flags"].any() and not payload["terminations"].any(),
        "Synthetic fixture cannot assert intervention or task termination",
    )
    require(
        torch.count_nonzero(payload["rewards"]).item() == 0,
        "Expected unscored zero reward",
    )
    expected = torch.zeros((count, 1, 1), dtype=torch.bool)
    expected[-1] = True
    require(
        torch.equal(payload["dones"], expected)
        and torch.equal(payload["truncations"], expected),
        "Expected final-horizon done/truncated only",
    )
    for key in ("curr_obs", "next_obs"):
        obs = payload[key]
        require(
            set(obs) == {"states", "main_images", "extra_view_images"},
            "Unexpected observation fields",
        )
        require(
            obs["states"].shape == (count, 1, 42)
            and obs["states"].dtype == torch.float32
            and torch.isfinite(obs["states"]).all().item(),
            "Expected finite float32 state [T,1,42]",
        )
        main, extra = obs["main_images"], obs["extra_view_images"]
        require(
            main.ndim == 5
            and main.shape[:2] == (count, 1)
            and main.shape[-1] == 3
            and min(main.shape[2:4]) > 0
            and main.dtype == torch.uint8,
            "Expected uint8 RGB HWC main images",
        )
        require(
            extra.shape == (count, 1, 1, *main.shape[2:])
            and extra.dtype == torch.uint8,
            "Expected independently associated uint8 D405 view [T,1,1,H,W,3]",
        )
    for key in payload["curr_obs"]:
        require(
            torch.equal(payload["curr_obs"][key][1:], payload["next_obs"][key][:-1]),
            f"Observation transition chain differs: {key}",
        )


def _robot(record: dict, part: str) -> None:
    state, status = record[part]["state"], record[part]["ros2"]
    fields = (
        list(STATE_SPEC)[:5] + ["tcp_force", "tcp_pose", "tcp_torque"]
        if part == "arm"
        else ["knuckle_position_rad", "normalized_closure"]
    )
    require(
        set(state) == set(fields),
        f"Expected exact {part} state fields; found {sorted(state)}",
    )
    for key in fields:
        size = STATE_SPEC[key][0]
        _numeric(state[key] if size > 1 else [state[key]], size, key)
    require(
        status["complete"] is True
        and not any(
            status[k]
            for k in (
                "missing_fields",
                "missing_streams",
                "stale_streams",
                "invalid_streams",
            )
        ),
        f"Incomplete or stale {part} measurements",
    )
    streams = status["streams"]
    if part == "arm":
        require(
            set(streams) == {"joints", "pose", "wrench_base", "wrench_stiffness"},
            "Missing FR3 stream",
        )
        joints = streams["joints"]["sample"]["data"]
        require(joints["names"] == list(FR3_JOINTS), "Incorrect FR3 joint names/order")
        for source, target in (
            ("position", "arm_joint_position"),
            ("velocity", "arm_joint_velocity"),
            ("effort", "arm_joint_effort"),
        ):
            _numeric(joints[source], 7, source)
            require(joints[source] == state[target], f"Decoded joint {source} differs")
        pose = streams["pose"]["sample"]["data"]["tcp_pose"]
        require(pose == state["tcp_pose"], "Pose stream/state mismatch")
        require(
            math.isclose(float(np.linalg.norm(pose[3:])), 1.0, abs_tol=1e-5),
            "Expected unit xyzw quaternion",
        )
        for stream, prefix in (("wrench_base", "base"), ("wrench_stiffness", "tcp")):
            data = streams[stream]["sample"]["data"]
            require(
                all(data[k] == state[f"{prefix}_{k}"] for k in ("force", "torque")),
                "Wrench stream/state mismatch",
            )
        frames = {
            "joints": "synthetic/base_link",
            "pose": "synthetic/base",
            "wrench_base": "synthetic/base",
            "wrench_stiffness": "synthetic/fr3_link2",
        }
        require(
            set(status["unavailable_fields"]) == {"tcp_vel", "arm_jacobian"},
            "Unexpected FR3 availability claim",
        )
    else:
        require(set(streams) == {"knuckle", "normalized"}, "Missing gripper stream")
        for stream, joint, key in (
            ("knuckle", "robotiq_85_left_knuckle_joint", "knuckle_position_rad"),
            ("normalized", "robotiq_normalized_joint", "normalized_closure"),
        ):
            data = streams[stream]["sample"]["data"]
            require(
                data["names"] == [joint] and data["position"] == [state[key]],
                "Incorrect gripper joint/state identity",
            )
        require(
            0 <= state["normalized_closure"] <= 1, "Normalized closure outside [0,1]"
        )
        frames = dict.fromkeys(streams, "synthetic/base_link")
        require(
            set(status["unavailable_fields"])
            == {"opening_width_m", "measured_velocity", "grip_force", "typed_status"},
            "Unqualified gripper measurement asserted",
        )
    for name, stream in streams.items():
        require(
            stream["fresh"] is True
            and stream["connected"] is True
            and stream["error"] is None,
            f"Unavailable measurement: {part}/{name}",
        )
        require(
            math.isfinite(stream["age_s"])
            and 0 <= stream["age_s"] <= status["max_age_s"],
            "Invalid reported observation freshness",
        )
        require(
            stream["sample"]["header"]["frame_id"] == frames[name],
            f"Wrong coordinate frame: {part}/{name}",
        )


def _camera(record: dict, name: str) -> None:
    require(
        set(record["cameras"]) == {"d405", "d435i"}, "Missing synthetic camera frame"
    )
    camera = record["cameras"][name]
    metadata = camera["metadata"]
    require(
        camera["dtype"] == "uint8"
        and len(camera["shape"]) == 3
        and camera["shape"][-1] == 3,
        "Expected camera uint8 HWC with three channels",
    )
    require(
        metadata["encoding"] == "RGB"
        and metadata["layout"] == "HWC"
        and metadata["source"] == f"fixture:{name}"
        and metadata["source_timestamp"] is None,
        "Incorrect camera encoding/layout/source timestamp provenance",
    )
    require(
        (camera["tensor_key"], camera["view_index"])
        == (("main_images", None) if name == "d435i" else ("extra_view_images", 0)),
        "Incorrect camera tensor slot",
    )


def _spacemouse(record: dict) -> None:
    mouse = record["spacemouse"]
    _numeric(mouse["axes"], 6, "SpaceMouse axes")
    require(
        len(mouse["buttons"]) == 2 and all(type(x) is bool for x in mouse["buttons"]),
        "Expected two SpaceMouse buttons",
    )
    metadata = mouse["metadata"]
    require(
        metadata["action_mapping"] is None
        and metadata["source"] == "fixture:spacemouse"
        and metadata["source_timestamp"] is None,
        "Unqualified SpaceMouse transformation/source claim",
    )


def _timing(
    record: dict,
    manifest: dict,
    boundary: int,
    frame: int,
    receipt_limit: int,
    receipt_floor: int,
    previous: dict,
    offsets: dict,
) -> None:
    timings, receipts = {}, {}
    for part in ("arm", "gripper"):
        for name, stream in record[part]["ros2"]["streams"].items():
            sample = stream["sample"]
            timing = sample["fixture_timing"]
            require(
                part not in timings or timing == timings[part],
                "Inconsistent robot stream times",
            )
            timings[part] = timing
            header = sample["header"]
            require(
                type(header["sec"]) is int
                and type(header["nanosec"]) is int
                and 0 <= header["nanosec"] < 1_000_000_000,
                "Invalid source header timestamp",
            )
            require(
                sample["source_epoch_offset_ns"] == SYNTHETIC_EPOCH_NS
                and sample["source_clock_domain"]
                == "synthetic-wall:" + manifest["synthetic_clock_domain"]
                and header["sec"] * 1_000_000_000 + header["nanosec"]
                == SYNTHETIC_EPOCH_NS + timing["source_ns"],
                "Incorrect synthetic header clock/epoch",
            )
            require(
                sample["receipt_clock_domain"] == manifest["recorder_clock_domain"],
                "Incompatible robot receipt clock",
            )
            receipts[f"{part}/{name}"] = _ns(sample["received_monotonic_ns"])
            sequence_key = f"sequence:{part}/{name}"
            sequence = stream["sequence"]
            require(type(sequence) is int and sequence > 0, "Missing stream sequence")
            if sequence_key in previous:
                require(
                    sequence == previous[sequence_key] + 1,
                    "Duplicate/out-of-order stream sequence",
                )
            previous[sequence_key] = sequence
    for name in ("d405", "d435i", "spacemouse"):
        metadata = (
            record["spacemouse"]["metadata"]
            if name == "spacemouse"
            else record["cameras"][name]["metadata"]
        )
        require(
            metadata["synthetic"] is True
            and metadata["clock_domain"] == manifest["recorder_clock_domain"],
            "Incompatible fixture receipt clock",
        )
        timings[name] = metadata["fixture_timing"]
        receipts[name] = _ns(metadata["receipt_monotonic_ns"])
    require(
        set(timings) == {"arm", "gripper", "d405", "d435i", "spacemouse"},
        "Missing timestamp stream",
    )
    for name, t in timings.items():
        require(
            t["synthetic"] is True
            and t["clock_domain"] == manifest["synthetic_clock_domain"],
            "Incompatible synthetic clock domain",
        )
        require(
            t["frame_index"] == frame and t["snapshot_ns"] == boundary,
            "Incorrect frame ordering/boundary",
        )
        for field in ("source_ns", "arrival_ns", "snapshot_ns"):
            _ns(t[field])
        require(
            t["missing"] is False and t["substituted"] is False,
            f"Missing/substituted frame: {name}",
        )
        if name in previous:
            require(
                t["source_ns"] > previous[name],
                f"Duplicate/out-of-order timestamp: {name}",
            )
        previous[name] = t["source_ns"]
    validate_alignment(
        timings,
        min(MAX_AGE_NS, manifest["alignment"]["max_age_ns"]),
        min(MAX_SKEW_NS, manifest["alignment"]["max_skew_ns"]),
    )
    for name, receipt in receipts.items():
        require(
            receipt_floor <= receipt <= receipt_limit,
            f"Observation receipt outside associated action/recorder interval: {name}",
        )
        previous_receipt = previous.get("receipt:" + name, 0)
        require(
            receipt > previous_receipt,
            f"Duplicate/out-of-order receipt timestamp: {name}",
        )
        previous["receipt:" + name] = receipt
        offsets["recorder_minus_receipt_ns/" + name].append(receipt_limit - receipt)
    for name, t in timings.items():
        offsets[f"source_age_ns/{name}"].append(boundary - t["source_ns"])
        offsets[f"arrival_age_ns/{name}"].append(boundary - t["arrival_ns"])
        offsets[f"transport_delay_ns/{name}"].append(t["arrival_ns"] - t["source_ns"])
        offsets[f"source_minus_arm_ns/{name}"].append(
            t["source_ns"] - timings["arm"]["source_ns"]
        )
    sources = [t["source_ns"] for t in timings.values()]
    offsets["cross_stream_skew_ns"].append(max(sources) - min(sources))


def _action(
    requested: dict,
    outcome: dict,
    episode: dict,
    manifest: dict,
    step: int,
    offsets: dict,
) -> None:
    _gate(requested)
    _gate(outcome)
    require(
        requested["schema"] == outcome["schema"] == PROVENANCE_SCHEMA
        and requested["event"] == "step_requested"
        and outcome["event"] == "step_outcome"
        and requested["step_index"] == outcome["step_index"] == step,
        "Duplicate/incorrect transition identifier/schema",
    )
    require(
        outcome["valid"] is True,
        "Rejected transition cannot be a successful demonstration",
    )
    require(
        requested["recorder_clock_domain"]
        == outcome["recorder_clock_domain"]
        == manifest["recorder_clock_domain"],
        "Incompatible recorder clock",
    )
    require(
        requested["synthetic_clock_domain"] == manifest["synthetic_clock_domain"],
        "Incompatible request logical clock",
    )
    require(
        episode["start_recorder_ns"]
        <= _ns(requested["recorder_ns"])
        <= _ns(outcome["recorder_ns"])
        <= episode["end_recorder_ns"],
        "Recorder event outside episode",
    )
    require(
        outcome["physical_controller_acceptance"] == "UNAVAILABLE"
        and outcome["physical_grasp_success"] is None,
        "Physical acceptance/grasp claim",
    )
    operator = requested["operator_request"]
    require(
        operator["source_id"].startswith("synthetic:")
        and operator["physical_mapping_verified"] is False
        and operator["interpretation"]
        == "scripted_fixture_request_v1; physical device mapping unverified"
        and operator["arm_kind"] == FR3_KIND
        and operator["gripper_kind"] == ROBOTIQ_KIND,
        "Missing/unqualified operator interpretation/source",
    )
    require(
        operator["raw_input"] == requested["observation_before"]["spacemouse"]
        and operator["input_timing"]
        == operator["raw_input"]["metadata"]["fixture_timing"],
        "Input/request timing binding differs",
    )
    for name, kind in (("arm", FR3_KIND), ("end_effector", ROBOTIQ_KIND)):
        request, events = (
            requested["typed_requests"][name],
            outcome["action_events"][name],
        )
        require(
            request["kind"] == kind
            and request["sequence"] == step + 1
            and request["source_id"] == operator["source_id"],
            "Wrong action discriminator/sequence/source",
        )
        require(
            request["session_id"] == episode["sessions"][name]["session_id"]
            and request["authority_id"] == episode["sessions"][name]["authority_id"],
            "Wrong action authority/session",
        )
        mock = outcome["mock_states"][name]
        require(
            mock["schema_version"] == 1
            and mock["offline"] is True
            and mock["physical_execution"] is False
            and mock["ros_domain_id"] is None,
            "Invalid mock status execution claim",
        )
        require(
            mock["session_id"] == request["session_id"]
            and mock["authority_id"] == request["authority_id"]
            and mock["clock_domain"] == request["clock_domain"]
            and mock["last_admitted_sequence"] == step + 1,
            "Mock status identity/sequence differs from typed journal",
        )
        require(
            mock["last_disposition"]
            in (
                ("acceptance_unknown", "mock_canceled")
                if name == "arm"
                else ("mock_completed",)
            ),
            "Rejected/failed mock status cannot be a successful demonstration",
        )
        require(
            request["clock_domain"] == manifest["recorder_clock_domain"],
            "Incompatible action clock domain",
        )
        part = "arm" if name == "arm" else "gripper"
        require(
            _parse_json(request["observation"]["snapshot_json"])
            == requested["observation_before"][part],
            "Action observation dependency differs",
        )
        require(
            request["interface"] == manifest["interfaces"][name],
            "Action interface identity differs",
        )
        require(
            request["values"]
            == (
                operator["arm_values"]
                if name == "arm"
                else [operator["normalized_closure"]]
            ),
            "Operator requested values differ from validated action",
        )
        profile = _validate_action(request, events, step)
        validate_observation(
            ObservationProvenance.from_observation(
                outcome["observation_after"][part],
                clock_domain=request["clock_domain"],
                timing_uncertainty_ns=0,
            ),
            kind,
            profile,
            request["clock_domain"],
            outcome["recorder_ns"],
        )
        require(
            requested["recorder_ns"]
            <= request["generated_ns"]
            <= outcome["recorder_ns"],
            "Request generation outside transition",
        )
        stages = [e["state"] for e in events]
        require(
            stages[:4] == ["created", "validated", "admitted", "submitted"],
            "Incorrect action stage ordering",
        )
        for event in events:
            require(
                event["request"] == request, "Action journal request identity changed"
            )
            require(
                request["generated_ns"]
                <= event["timestamp_ns"]
                <= outcome["recorder_ns"],
                "Action event outside transition",
            )
            if event["state"] in ("validated", "admitted", "submitted"):
                require(
                    event["timestamp_ns"] < request["deadline_ns"],
                    "Expired action admitted/submitted",
                )
        submitted_ns = next(
            e["timestamp_ns"] for e in events if e["state"] == "submitted"
        )
        offsets[f"request_to_submission_ns/{name}"].append(
            submitted_ns - request["generated_ns"]
        )
        offsets[f"submission_to_outcome_ns/{name}"].append(
            outcome["recorder_ns"] - submitted_ns
        )
        for stream, status in requested["observation_before"][part]["ros2"][
            "streams"
        ].items():
            offsets[f"receipt_age_at_request_ns/{part}/{stream}"].append(
                request["generated_ns"] - status["sample"]["received_monotonic_ns"]
            )


def distribution(values: list[int]) -> dict:
    """Summarize observed offsets, retaining the complete small fixture distribution."""
    a = np.asarray(values, dtype=np.int64)
    return {
        "count": len(values),
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "histogram": {str(k): v for k, v in sorted(Counter(values).items())},
    }


def _rejected(outcome: dict) -> bool:
    if outcome.get("valid") is False:
        return True
    rejected = ("rejected", "expired", "failed", "mock_rejected", "mock_failed")
    events, snapshots = outcome.get("action_events"), outcome.get("mock_states")
    if isinstance(events, dict):
        for journal in events.values():
            if isinstance(journal, list) and any(
                isinstance(e, dict) and e.get("state") in rejected for e in journal
            ):
                return True
    return isinstance(snapshots, dict) and any(
        isinstance(s, dict) and s.get("last_disposition") in rejected
        for s in snapshots.values()
    )


class _Audit:
    def __init__(self):
        self.errors = []
        self.checks = Counter()
        self.failures = Counter()

    def run(self, category: str, path: str, call: Callable[[], Any]) -> Any:
        self.checks[category] += 1
        try:
            return call()
        except Exception as exc:
            self.failures[category] += 1
            self.errors.append(
                {
                    "category": category,
                    "path": path,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return None


def downstream_gates() -> dict:
    """Return source-verified blockers without constructing a model or training data."""
    return {
        "openpi_pi05": {
            "status": "UNQUALIFIED",
            "gaps": [
                "checkpoint_and_training_config_unselected",
                "physical_camera_roles_and_calibration_unknown",
                "state_42_incompatible_with_realworld_input_19",
                "normalized_offset_semantics_not_checkpoint_verified",
                "action_horizon_and_boundary_padding_unselected",
                "normalization_statistics_unqualified",
                "task_instruction_missing_from_native_tensors",
                "fr3_preprocessing_transform_missing",
            ],
        },
        "recap": {
            "status": "SYNTHETIC_PIPELINE_ONLY",
            "missing": [
                "task_identity_and_instruction",
                "policy_rollout_id",
                "policy_checkpoint_version",
                "real_operator_interventions",
                "physical_controller_acceptance",
                "evaluated_outcomes_and_success_annotations",
                "train_eval_split",
                "qualified_rewards_returns_advantages",
            ],
        },
    }


def assess_synthetic_dataset(root: Path | str) -> dict:
    """Audit native artifacts without mutation; return diagnostics even on invalid input.

    Episode results describe inspected artifacts. Dataset-wide integrity errors still
    block export even when individual episodes have valid schemas.
    """
    root = Path(root).absolute()
    audit = _Audit()
    report = {
        "schema": QUALITY_SCHEMA,
        "validator_version": VALIDATOR_VERSION,
        **GATE,
        "dataset_id": None,
        "source_manifest_sha256": None,
        "episodes": [],
        "counts": {},
        "timestamp_statistics": {},
        "missing_features": {},
        "downstream": downstream_gates(),
        "export": {"status": "NOT_REQUESTED"},
    }
    manifest_path = audit.run(
        "integrity", "manifest.json", lambda: artifact(root, "manifest.json")
    )
    manifest = (
        audit.run("schema", "manifest.json", lambda: read_json(manifest_path))
        if manifest_path
        else None
    )
    if manifest_path:
        report["source_manifest_sha256"] = file_digest(manifest_path)
    if not isinstance(manifest, dict):
        audit.run(
            "schema",
            "manifest.json",
            lambda: require(False, "Expected manifest JSON object"),
        )
        return _finish(report, audit, {})
    report["dataset_id"] = manifest.get("dataset_id")
    audit.run("schema", "manifest.json", lambda: _manifest(manifest))
    period = (
        audit.run(
            "schema",
            "manifest.json/step_period_ns",
            lambda: _ns(manifest["step_period_ns"]),
        )
        or 0
    )
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        files = {}
    actual_files = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    report["unindexed_artifacts"] = sorted(actual_files - set(files))
    audit.run(
        "integrity",
        "manifest.json/files",
        lambda: require(
            actual_files == set(files), "Unindexed artifact or missing inventory entry"
        ),
    )
    bad_files = set()
    for relative, digest in files.items():

        def verify():
            require(
                file_digest(artifact(root, relative)) == digest,
                f"Corrupt artifact/hash mismatch: {relative}",
            )
            return True

        if not audit.run("integrity", relative, verify):
            bad_files.add(relative)
    index = audit.run(
        "schema",
        "demos/trajectory_index.json",
        lambda: read_json(artifact(root, "demos/trajectory_index.json")),
    )
    entries = manifest.get("episodes", [])
    if not isinstance(entries, list):
        entries = []
    entries = list(entries)
    indexed_metadata = {
        e["metadata_path"]
        for e in entries
        if isinstance(e, dict) and isinstance(e.get("metadata_path"), str)
    }
    # Interrupted collection can leave an episode before its manifest entry exists.
    for path in sorted(root.glob("episodes/*/episode.json")):
        relative = str(path.relative_to(root))
        if relative not in indexed_metadata:
            audit.run(
                "episode",
                relative,
                lambda: require(False, "Unindexed episode retained for audit"),
            )
            entries.append(
                {
                    "metadata_path": relative,
                    "provenance_path": str(
                        path.with_name("steps.jsonl").relative_to(root)
                    ),
                }
            )
    ids, environment_ids, previous_episode = set(), set(), None
    offsets = defaultdict(list)
    provenance_counts = Counter()
    for number, entry in enumerate(entries):
        path = f"episodes/{number}"
        start_errors = len(audit.errors)
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(k), str) for k in ("metadata_path", "provenance_path")
        ):
            audit.run(
                "schema",
                path,
                lambda: require(
                    False, "Expected episode metadata/provenance path strings"
                ),
            )
            report["episodes"].append(
                {
                    "index": number,
                    "episode_id": None,
                    "status": "CORRUPTED_ARTIFACT",
                    "transitions": [],
                    "diagnostics": audit.errors[start_errors:],
                }
            )
            continue
        episode = audit.run(
            "episode", path, lambda: read_json(artifact(root, entry["metadata_path"]))
        )
        result = {
            "index": number,
            "episode_id": episode.get("episode_id")
            if isinstance(episode, dict)
            else None,
            "status": "INVALID",
            "transitions": [],
            "diagnostics": [],
        }
        report["episodes"].append(result)
        if not isinstance(episode, dict):
            result["status"] = "CORRUPTED_ARTIFACT"
            result["diagnostics"] = audit.errors[start_errors:]
            continue
        audit.run(
            "episode",
            path,
            lambda: _episode(episode, manifest, ids, environment_ids, previous_episode),
        )
        previous_episode = episode
        records = audit.run(
            "provenance",
            path,
            lambda: load_records(artifact(root, entry["provenance_path"])),
        )
        native = audit.run("integrity", path, lambda: native_path(root, index, number))
        payload = (
            audit.run(
                "serialization",
                path,
                lambda: _load_native(native),
            )
            if native
            else None
        )
        count = episode.get("step_count", 0)
        if type(count) is not int or not 1 <= count <= 1000:
            count = 0
        tensor_valid = isinstance(payload, dict) and bool(count)
        if tensor_valid:
            tensor_error_start = len(audit.errors)
            audit.run("tensors", path, lambda: _tensor_contract(payload, count))
            audit.run(
                "serialization",
                path,
                lambda: _index_contract(index, payload, number, count),
            )
            tensor_valid = len(audit.errors) == tensor_error_start
        previous = {}
        if records is not None:
            audit.run(
                "episode",
                path,
                lambda: require(
                    len(records) == 2 * count, "Incomplete request/outcome record count"
                ),
            )
            for step in range((len(records) + 1) // 2):
                location = f"{path}/steps/{step}"
                before_errors = len(audit.errors)
                audit.run(
                    "tensors",
                    location,
                    lambda: require(
                        tensor_valid,
                        "Native transition payload unavailable or tensor contract invalid",
                    ),
                )
                pair = records[2 * step : 2 * step + 2]
                if len(pair) != 2:
                    audit.run(
                        "provenance",
                        location,
                        lambda: require(
                            False, "Incomplete transition: missing outcome"
                        ),
                    )
                else:
                    request, outcome = pair
                    event_sets = outcome.get("action_events", {})
                    if isinstance(event_sets, dict):
                        for events in event_sets.values():
                            if isinstance(events, list):
                                provenance_counts.update(
                                    e.get("state", "missing_state")
                                    for e in events
                                    if isinstance(e, dict)
                                    and isinstance(e.get("state"), str)
                                )
                    audit.run(
                        "actions",
                        location,
                        lambda: _action(
                            request, outcome, episode, manifest, step, offsets
                        ),
                    )
                    if step:
                        audit.run(
                            "timing",
                            location,
                            lambda: require(
                                request["recorder_ns"]
                                >= records[2 * step - 1]["recorder_ns"],
                                "Out-of-order recorder transitions",
                            ),
                        )
                    logical = (
                        audit.run(
                            "timing",
                            location,
                            lambda: _ns(episode["start_logical_ns"]) + step * period,
                        )
                        or 0
                    )
                    audit.run(
                        "timing",
                        location,
                        lambda: require(
                            request["logical_action_ns"] == logical,
                            "Wrong logical action timestamp",
                        ),
                    )
                    for key, obs_key, frame, boundary, event in (
                        ("observation_before", "curr_obs", step, logical, request),
                        (
                            "observation_after",
                            "next_obs",
                            step + 1,
                            logical + period,
                            outcome,
                        ),
                    ):
                        obs = event.get(key, {})
                        for category, check in (
                            ("arm", lambda: _robot(obs, "arm")),
                            ("gripper", lambda: _robot(obs, "gripper")),
                            ("d405", lambda: _camera(obs, "d405")),
                            ("d435i", lambda: _camera(obs, "d435i")),
                            ("spacemouse", lambda: _spacemouse(obs)),
                        ):
                            audit.run(category, location + "/" + key, check)
                        if isinstance(payload, dict):
                            audit.run(
                                "association",
                                location + "/" + key,
                                lambda: _validate_observation(
                                    obs,
                                    payload[obs_key],
                                    step,
                                    frame,
                                    boundary,
                                    manifest,
                                ),
                            )
                        if key == "observation_before" and step:
                            audit.run(
                                "association",
                                location,
                                lambda: require(
                                    obs == records[2 * step - 1]["observation_after"],
                                    "Source observation chain changed",
                                ),
                            )
                        else:
                            audit.run(
                                "timing",
                                location + "/" + key,
                                lambda: _timing(
                                    obs,
                                    manifest,
                                    boundary,
                                    frame,
                                    event["recorder_ns"],
                                    episode["start_recorder_ns"]
                                    if key == "observation_before"
                                    else max(
                                        r["generated_ns"]
                                        for r in request["typed_requests"].values()
                                    ),
                                    previous,
                                    offsets,
                                ),
                            )
                    if isinstance(payload, dict):
                        audit.run(
                            "association",
                            location,
                            lambda: _dense_action(request, payload, step),
                        )
                rejected = len(pair) == 2 and _rejected(pair[1])
                result["transitions"].append(
                    {
                        "step_index": step,
                        "status": "REJECTED"
                        if rejected
                        else "INVALID"
                        if len(audit.errors) > before_errors
                        else "VALID_SYNTHETIC",
                        "diagnostics": audit.errors[before_errors:],
                    }
                )
        associated = {
            entry.get("metadata_path"),
            entry.get("provenance_path"),
            str(native.relative_to(root)) if native else None,
        }
        result["diagnostics"] = audit.errors[start_errors:]
        if episode.get("status") != "COMPLETE":
            result["status"] = "INCOMPLETE"
        elif (
            native is None
            or payload is None
            or bad_files.intersection(associated)
            or records is None
        ):
            result["status"] = "CORRUPTED_ARTIFACT"
        elif len(records) != 2 * count:
            result["status"] = "INCOMPLETE"
        elif len(audit.errors) == start_errors:
            result["status"] = "VALID_SYNTHETIC"
    # The native loader is used only after validating the paths and bounded tensor schema.
    if not audit.errors:
        audit.run("native_reload", "demos", lambda: validate_synthetic_dataset(root))
    report["missing_features"] = {
        "physical_camera_roles": 2,
        "camera_calibrations": 2,
        "recoverable_resolved_config": 1,
        "task_instruction": len(entries),
        "physical_device_timestamps_camera_and_input": 3
        * sum(len(e["transitions"]) + 1 for e in report["episodes"]),
        "physical_controller_acceptance": sum(
            len(e["transitions"]) for e in report["episodes"]
        ),
    }
    report["action_provenance"] = {
        "event_counts": dict(sorted(provenance_counts.items())),
        "physical_execution": False,
        "physical_controller_acceptance": "UNAVAILABLE",
        "physical_grasp_success": None,
    }
    report["observation_contract"] = {
        k: {"dimension": dim, "units": units} for k, (dim, units) in STATE_SPEC.items()
    }
    return _finish(report, audit, offsets)


def _manifest(m: dict) -> None:
    _gate(m)
    require(
        m["schema"] == SCHEMA and m["provenance_schema"] == PROVENANCE_SCHEMA,
        "Unsupported dataset/schema version",
    )
    require(m["status"] == "COMPLETE" and not m["errors"], "Incomplete dataset status")
    require(
        m["action_schema_version"] == m["observation_schema_version"] == 1,
        "Unsupported action/observation schema version",
    )
    require(m["state_fields"] == list(STATE_SPEC), "Unexpected state ordering")
    require(
        m["action_kinds"] == [FR3_KIND, ROBOTIQ_KIND],
        "Wrong dataset action discriminators",
    )
    require(
        m["synthetic_clock_domain"] == "synthetic:" + m["dataset_id"]
        and m["recorder_clock_domain"].startswith("local-monotonic:"),
        "Invalid clock identities",
    )
    require(
        type(m["expected_episodes"]) is int
        and 1 <= m["expected_episodes"] <= 100
        and isinstance(m["episodes"], list)
        and len(m["episodes"]) == m["expected_episodes"],
        "Incomplete episode inventory",
    )
    require(
        type(m["expected_steps_per_episode"]) is int
        and 1 <= m["expected_steps_per_episode"] <= 1000,
        "Invalid step count",
    )
    require(
        type(m["step_period_ns"]) is int and m["step_period_ns"] > 0,
        "Invalid logical period",
    )
    require(
        0 < m["alignment"]["max_age_ns"] <= MAX_AGE_NS
        and 0 < m["alignment"]["max_skew_ns"] <= MAX_SKEW_NS,
        "Unqualified synthetic timing tolerance",
    )
    require(
        _ns(m["started_wall_ns"]) <= _ns(m["ended_wall_ns"]),
        "Dataset wall time reversal",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", m["config_sha256"]) is not None
        and re.fullmatch(r"[0-9a-f]{64}", m["recorder_software"]["source_sha256"])
        is not None,
        "Invalid configuration/software digest",
    )


def _episode(
    e: dict, m: dict, ids: set, environment_ids: set, previous: dict | None
) -> None:
    _gate(e)
    require(
        e["episode_id"] not in ids
        and e["environment_episode_id"] not in environment_ids,
        "Duplicate episode identity",
    )
    ids.add(e["episode_id"])
    environment_ids.add(e["environment_episode_id"])
    require(
        e["schema"] == PROVENANCE_SCHEMA and e["dataset_id"] == m["dataset_id"],
        "Episode schema/dataset mismatch",
    )
    require(
        e["status"] == "COMPLETE"
        and e["step_count"] == m["expected_steps_per_episode"],
        "Incomplete episode",
    )
    require(
        e["termination_reason"] == "synthetic_horizon"
        and e["reward_evaluated"] is False
        and e["success_evaluated"] is False
        and e["reward_semantics"] == "unscored_fixture_zero"
        and e["intervention_semantics"] == "no_human_or_policy_intervention",
        "Unqualified outcome/reward/intervention label",
    )
    require(
        e["recorder_clock_domain"] == m["recorder_clock_domain"]
        and _ns(e["start_recorder_ns"]) < _ns(e["end_recorder_ns"]),
        "Invalid episode recorder clock",
    )
    require(
        e["end_logical_ns"]
        == e["start_logical_ns"] + e["step_count"] * m["step_period_ns"],
        "Wrong episode logical duration",
    )
    if previous:
        require(
            e["start_logical_ns"] == previous["end_logical_ns"] + m["step_period_ns"]
            and e["start_recorder_ns"] >= previous["end_recorder_ns"],
            "Overlapping/out-of-order episode boundaries",
        )
        require(
            all(
                e["sessions"][k] != previous["sessions"][k]
                for k in ("arm", "end_effector")
            ),
            "Reused action session across reset",
        )


def _index_contract(index: dict, payload: dict, number: int, count: int) -> None:
    info = index["trajectory_index"][str(number)]
    require(
        info["trajectory_id"] == number
        and info["num_samples"] == count
        and info["shape"] == [count, 1, 1]
        and info["max_episode_length"] == count
        and payload["max_episode_length"] == count
        and info["model_weights_id"] == payload["model_weights_id"],
        "Native trajectory/index identity mismatch",
    )


def _dense_action(request: dict, payload: dict, step: int) -> None:
    values = (
        request["typed_requests"]["arm"]["values"]
        + request["typed_requests"]["end_effector"]["values"]
    )
    require(
        torch.equal(
            torch.tensor(values, dtype=torch.float32), payload["actions"][step, 0]
        ),
        "Action tensor/provenance mismatch",
    )


def _finish(report: dict, audit: _Audit, offsets: dict) -> dict:
    episodes = report["episodes"]
    transitions = [t for e in episodes for t in e["transitions"]]
    report["counts"] = {
        "episodes": len(episodes),
        "valid_episodes": sum(e["status"] == "VALID_SYNTHETIC" for e in episodes),
        "invalid_episodes": sum(e["status"] == "INVALID" for e in episodes),
        "incomplete_episodes": sum(e["status"] == "INCOMPLETE" for e in episodes),
        "corrupted_episodes": sum(
            e["status"] == "CORRUPTED_ARTIFACT" for e in episodes
        ),
        "transitions": len(transitions),
        "valid_transitions": sum(t["status"] == "VALID_SYNTHETIC" for t in transitions),
        "invalid_transitions": sum(
            t["status"] != "VALID_SYNTHETIC" for t in transitions
        ),
        "rejected_transitions": sum(t["status"] == "REJECTED" for t in transitions),
    }
    report["timestamp_statistics"] = {
        k: distribution(v) for k, v in sorted(offsets.items()) if v
    }
    report["timing_contract"] = {
        "units": "ns",
        "max_source_age_ns": MAX_AGE_NS,
        "max_cross_stream_skew_ns": MAX_SKEW_NS,
        "physical_budget_qualified": False,
        "sampling": "initial observation plus each next observation; shared references checked separately",
    }
    report["modalities"] = {
        k: {
            "checks": audit.checks[k],
            "failures": audit.failures[k],
            "status": "FAIL"
            if audit.failures[k]
            else "PASS"
            if audit.checks[k]
            else "NOT_CHECKED",
        }
        for k in (
            "arm",
            "gripper",
            "d405",
            "d435i",
            "spacemouse",
            "actions",
            "timing",
            "tensors",
            "association",
        )
    }
    report["diagnostics"] = audit.errors
    report["valid"] = not audit.errors
    report["readiness"] = {
        "native_rlinf_serialization": "PASS"
        if not audit.errors and audit.checks["native_reload"]
        else "FAIL",
        "multimodal_schema": "PASS" if not audit.errors else "FAIL",
        "synthetic_temporal_consistency": report["modalities"]["timing"]["status"],
        "typed_action_provenance": report["modalities"]["actions"]["status"],
        "lerobot_export": "NOT_REQUESTED",
        "openpi_pi05_schema": "UNQUALIFIED",
        "real_world_demonstration": "NOT_QUALIFIED",
    }
    return report
