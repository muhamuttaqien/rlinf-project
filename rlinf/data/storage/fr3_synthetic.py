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

"""Sidecar integrity and native replay validation for synthetic FR3 datasets."""

import hashlib
import json
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.robotics.parts.fr3_fixtures import validate_alignment
from rlinf.robotics.parts.transports.ros2.action_contract import (
    FR3_KIND,
    ROBOTIQ_KIND,
    ActionRequest,
    ActionState,
    InterfaceIdentity,
    MockAuthorization,
    MockLimitProfile,
    ObservationProvenance,
    validate_observation,
    validate_request,
)

SCHEMA = "fr3_synthetic_dataset_v1"
PROVENANCE_SCHEMA = "fr3_synthetic_provenance_v1"
GATE = {
    "synthetic": True,
    "physical_execution": False,
    "qualification_gate": "NO_GO",
    "physical_command_authorization": "NOT_GRANTED",
    "physical_safety_qualification": "NOT_QUALIFIED",
    "eligible_real_demonstration": False,
}


def json_value(value: Any) -> Any:
    """Convert fixture values to finite JSON, retaining invalid numbers as text."""
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path: Path, value: dict) -> None:
    """Replace one owned status file only after flushing its complete JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(json_value(value), stream, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_record(path: Path, value: dict) -> None:
    """Persist an attempt or outcome before advancing the collection loop."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(json_value(value), allow_nan=False, sort_keys=True) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def file_digest(path: Path) -> str:
    """Hash a file without retaining the dataset in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_json(text: str) -> Any:

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Invalid JSON constant {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def read_json(path: Path) -> Any:
    """Read strict JSON; duplicate keys and nonfinite constants are invalid."""
    return _parse_json(path.read_text())


def observation_record(raw: dict) -> dict:
    """Associate native tensor slots with their source snapshot and pixel hashes."""
    result = {
        "arm": raw["arm_observation"],
        "gripper": raw["gripper_observation"],
        "spacemouse": raw["spacemouse"],
        "cameras": {},
    }
    for name, tensor_key, view in (
        ("d435i", "main_images", None),
        ("d405", "extra_view_images", 0),
    ):
        image = raw[name]["image"]
        result["cameras"][name] = {
            "metadata": raw[name]["metadata"],
            "tensor_key": tensor_key,
            "view_index": view,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        }
    return json_value(result)


def _check(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError("Invalid synthetic dataset: " + detail)


def _validate_action(
    request_data: dict, events: list[dict], last_sequence: int
) -> MockLimitProfile:
    validated = [e for e in events if e["state"] == "validated"]
    _check(len(validated) == 1, "expected one validation event per request")
    event = validated[0]
    _check(event["request"] == request_data, "request differs from validated action")
    interface = InterfaceIdentity(
        **{
            **request_data["interface"],
            "mounting_rpy_rad": tuple(request_data["interface"]["mounting_rpy_rad"]),
        }
    )
    request = ActionRequest(
        **{
            **request_data,
            "values": tuple(request_data["values"]),
            "interface": interface,
            "observation": ObservationProvenance(**request_data["observation"]),
            "disposition": ActionState(request_data["disposition"]),
        }
    )
    profile_data = event["approval"]["profile"]
    profile = MockLimitProfile(
        **{
            **profile_data,
            "interface": interface,
            "lower": tuple(profile_data["lower"]),
            "upper": tuple(profile_data["upper"]),
        }
    )
    _check(
        profile_data["interface"] == request_data["interface"],
        "profile interface mismatch",
    )
    authorization = MockAuthorization(**event["approval"]["authorization"])
    validate_request(
        request,
        profile,
        authorization,
        interface,
        request.session_id,
        request.authority_id,
        request.clock_domain,
        last_sequence,
        event["timestamp_ns"],
    )
    states = [e["state"] for e in events]
    _check(
        all(s in states for s in ("created", "validated", "admitted", "submitted")),
        "missing action stage",
    )
    _check(
        not any(s in states for s in ("failed", "expired", "rejected")),
        "failed action in complete episode",
    )
    _check(
        all(
            e["schema_version"] == 1
            and e["offline"] is True
            and e["physical_execution"] is False
            for e in events
        ),
        "invalid event execution claim",
    )
    _check(
        all(e["clock_domain"] == request.clock_domain for e in events),
        "mixed event clocks",
    )
    times = [e["timestamp_ns"] for e in events]
    _check(times == sorted(times), "event time reversal")
    submission = [e for e in events if e["state"] == "submitted"]
    _check(len(submission) == 1, "expected one mock submission")
    values = request_data["values"]
    wire = (
        {
            "linear": dict(zip(("x", "y", "z"), values[:3])),
            "angular": dict(zip(("x", "y", "z"), values[3:])),
        }
        if request.kind == FR3_KIND
        else {"command": {"position": values[0], "max_effort": 0.0}}
    )
    _check(submission[0]["detail"]["wire"] == wire, "wrong mock message representation")
    if request.kind == ROBOTIQ_KIND:
        _check(
            "mock_acknowledged" in states and "mock_completed" in states,
            "gripper mock did not complete",
        )
    else:
        _check(
            request.kind == FR3_KIND and "acceptance_unknown" in states,
            "arm acceptance misrepresented",
        )
    return profile


def _validate_observation(
    record: dict,
    tensors: dict,
    index: int,
    frame_index: int,
    timestamp_ns: int,
    manifest: dict,
) -> None:
    from rlinf.envs.real.fr3.config import STATE_FIELDS

    fields = {**record["arm"]["state"], **record["gripper"]["state"]}
    state = torch.tensor(
        [
            v
            for k in STATE_FIELDS
            for v in (fields[k] if isinstance(fields[k], list) else [fields[k]])
        ],
        dtype=torch.float32,
    )
    _check(
        torch.equal(tensors["states"][index, 0], state),
        "dense state differs from provenance",
    )
    timing = {}
    for part, domain in (("arm", 70), ("gripper", 71)):
        status = record[part]["ros2"]
        _check(
            status["complete"] is True and status["domain_id"] == domain,
            "incomplete or wrong-domain observation",
        )
        for stream in status["streams"].values():
            sample = stream["sample"]
            _check(
                sample["synthetic"] is True
                and sample["source"] == "offline_synthetic_fixture",
                "non-synthetic sample",
            )
            _check(
                sample["header"]["frame_id"].startswith("synthetic/"),
                "unexpected frame identity",
            )
            if part in timing:
                _check(
                    timing[part] == sample["fixture_timing"],
                    "inconsistent arm/gripper stream timing",
                )
            timing[part] = sample["fixture_timing"]
            _check(
                sample["receipt_clock_domain"] == manifest["recorder_clock_domain"],
                "wrong receipt clock domain",
            )
            _check(
                sample["source_epoch_offset_ns"] + timing[part]["source_ns"]
                == sample["header"]["sec"] * 1_000_000_000
                + sample["header"]["nanosec"],
                "source header association mismatch",
            )
    for name, camera in record["cameras"].items():
        pixel = tensors[camera["tensor_key"]][index, 0]
        if camera["view_index"] is not None:
            pixel = pixel[camera["view_index"]]
        _check(
            list(pixel.shape) == camera["shape"] and pixel.dtype == torch.uint8,
            "image shape or dtype mismatch",
        )
        _check(
            hashlib.sha256(pixel.numpy().tobytes()).hexdigest() == camera["sha256"],
            "image association mismatch",
        )
        _check(camera["metadata"]["synthetic"] is True, "non-synthetic camera")
        timing[name] = camera["metadata"]["fixture_timing"]
    _check(
        record["spacemouse"]["metadata"]["synthetic"] is True
        and record["spacemouse"]["metadata"]["action_mapping"] is None,
        "physical input mapping asserted",
    )
    timing["spacemouse"] = record["spacemouse"]["metadata"]["fixture_timing"]
    _check(
        all(
            t["frame_index"] == frame_index
            and t["clock_domain"] == manifest["synthetic_clock_domain"]
            and t["snapshot_ns"] == timestamp_ns
            for t in timing.values()
        ),
        "wrong frame-to-step association",
    )
    validate_alignment(
        timing,
        manifest["alignment"]["max_age_ns"],
        manifest["alignment"]["max_skew_ns"],
    )


def _validate_dataset(root: Path, status: str) -> dict:
    from rlinf.data.storage.replay import TrajectoryReplayBuffer
    from rlinf.envs.real.fr3.config import CAMERAS, ENV_ID, STATE_FIELDS

    manifest = read_json(root / "manifest.json")
    _check(
        manifest["schema"] == SCHEMA and manifest["status"] == status,
        "schema/status is not finalized",
    )
    _check(all(manifest.get(k) == v for k, v in GATE.items()), "physical gate changed")
    _check(
        manifest["environment_id"] == ENV_ID and manifest["env_type"] == "fr3_offline",
        "environment identity mismatch",
    )
    _check(
        manifest["state_fields"] == list(STATE_FIELDS)
        and manifest["action_schema_version"]
        == manifest["observation_schema_version"]
        == 1,
        "unknown observation/action schema",
    )
    _check(
        manifest["camera_roles"] == {"d405": None, "d435i": None},
        "physical camera roles asserted",
    )
    _check(
        manifest["camera_identities"]
        == {k: {"model": v[0], "serial_number": v[1]} for k, v in CAMERAS.items()},
        "camera identity mismatch",
    )
    _check(
        manifest["provenance_schema"] == PROVENANCE_SCHEMA, "unknown provenance schema"
    )
    _check(
        manifest["action_kinds"] == [FR3_KIND, ROBOTIQ_KIND], "wrong action semantics"
    )
    _check(
        manifest["alignment"]["policy"] == "latest_before_action_no_substitution",
        "unknown alignment policy",
    )
    for relative, digest in manifest["files"].items():
        relative_path = Path(relative)
        _check(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            "unsafe artifact path",
        )
        target = root / relative_path
        _check(
            not any(p.is_symlink() for p in (target, *target.parents)),
            "symlink artifact",
        )
        _check(
            target.is_file() and file_digest(target) == digest,
            f"missing/corrupt artifact {relative}",
        )
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    _check(actual == set(manifest["files"]), "unindexed artifact or missing hash")
    index = read_json(root / "demos/trajectory_index.json")
    metadata = read_json(root / "demos/metadata.json")
    episodes = manifest["episodes"]
    _check(
        len(episodes) == manifest["expected_episodes"] and 1 <= len(episodes) <= 100,
        "incomplete episode set",
    )
    _check(
        index["trajectory_id_list"] == list(range(len(episodes))),
        "episode/trajectory index mismatch",
    )
    _check(
        metadata["trajectory_format"] == "pt" and metadata["size"] == len(episodes),
        "native metadata mismatch",
    )
    total = 0
    for trajectory_id, entry in enumerate(episodes):
        _check(
            entry["metadata_path"] in manifest["files"]
            and entry["provenance_path"] in manifest["files"],
            "unindexed episode sidecar",
        )
        episode = read_json(root / entry["metadata_path"])
        _check(
            episode["status"] == "COMPLETE" and episode["step_count"] > 0,
            "incomplete episode",
        )
        _check(
            all(episode.get(k) == v for k, v in GATE.items()), "episode execution claim"
        )
        _check(
            episode["termination_reason"] == "synthetic_horizon"
            and episode["success_evaluated"] is False,
            "invented task success",
        )
        count = episode["step_count"]
        _check(
            episode["schema"] == PROVENANCE_SCHEMA
            and episode["dataset_id"] == manifest["dataset_id"],
            "episode schema/session mismatch",
        )
        _check(
            count == manifest["expected_steps_per_episode"], "incomplete episode length"
        )
        _check(
            episode["recorder_clock_domain"] == manifest["recorder_clock_domain"]
            and episode["start_recorder_ns"] <= episode["end_recorder_ns"],
            "invalid recorder timing",
        )
        _check(
            episode["end_logical_ns"]
            == episode["start_logical_ns"] + count * manifest["step_period_ns"],
            "logical episode duration mismatch",
        )
        info = index["trajectory_index"][str(trajectory_id)]
        filename = (
            "demos/trajectory_"
            + str(trajectory_id)
            + "_"
            + info["model_weights_id"]
            + ".pt"
        )
        _check(filename in manifest["files"], "unindexed native trajectory")
        payload = torch.load(root / filename, map_location="cpu", weights_only=True)
        _check(
            payload["actions"].shape == (count, 1, 7)
            and payload["rewards"].shape == (count, 1, 1),
            "native action/reward dimensions",
        )
        _check(
            torch.isfinite(payload["actions"]).all().item()
            and torch.count_nonzero(payload["rewards"]).item() == 0,
            "invalid payload/reward",
        )
        _check(
            not payload["intervene_flags"].any().item(),
            "physical/expert interventions asserted",
        )
        _check(
            not payload["terminations"].any().item(),
            "physical task termination asserted",
        )
        expected = torch.zeros((count, 1, 1), dtype=torch.bool)
        expected[-1] = True
        _check(
            torch.equal(payload["truncations"], expected)
            and torch.equal(payload["dones"], expected),
            "episode boundaries",
        )
        lines = (root / entry["provenance_path"]).read_text().splitlines(keepends=True)
        _check(
            len(lines) == count * 2 and all(line.endswith("\n") for line in lines),
            "partial step log",
        )
        records = [_parse_json(line) for line in lines]
        for i in range(count):
            requested, outcome = records[2 * i : 2 * i + 2]
            _check(
                requested["event"] == "step_requested"
                and outcome["event"] == "step_outcome"
                and outcome["valid"] is True,
                "incomplete request/outcome",
            )
            _check(
                requested["step_index"] == outcome["step_index"] == i
                and requested["schema"] == outcome["schema"] == PROVENANCE_SCHEMA,
                "step schema/index",
            )
            _check(
                requested["physical_execution"] is False
                and outcome["physical_execution"] is False,
                "step physical execution claim",
            )
            values = []
            logical_ns = episode["start_logical_ns"] + i * manifest["step_period_ns"]
            _check(
                requested["logical_action_ns"] == logical_ns,
                "wrong logical action time",
            )
            _check(
                requested["recorder_clock_domain"]
                == outcome["recorder_clock_domain"]
                == manifest["recorder_clock_domain"],
                "mixed recorder clocks",
            )
            _check(
                episode["start_recorder_ns"]
                <= requested["recorder_ns"]
                <= outcome["recorder_ns"]
                <= episode["end_recorder_ns"],
                "recorder time reversal",
            )
            _check(
                outcome["physical_controller_acceptance"] == "UNAVAILABLE"
                and outcome["physical_grasp_success"] is None,
                "physical acceptance/grasp asserted",
            )
            operator = requested["operator_request"]
            _check(
                operator["raw_input"] == requested["observation_before"]["spacemouse"]
                and operator["physical_mapping_verified"] is False,
                "operator input/provenance mismatch",
            )
            for name, kind in (("arm", FR3_KIND), ("end_effector", ROBOTIQ_KIND)):
                request = requested["typed_requests"][name]
                _check(
                    request["interface"] == manifest["interfaces"][name],
                    "dataset interface identity mismatch",
                )
                _check(
                    request["kind"] == kind and request["sequence"] == i + 1,
                    "action schema or sequence",
                )
                _check(
                    request["session_id"] == episode["sessions"][name]["session_id"]
                    and request["authority_id"]
                    == episode["sessions"][name]["authority_id"],
                    "action authority epoch mismatch",
                )
                profile = _validate_action(request, outcome["action_events"][name], i)
                part = "arm" if name == "arm" else "gripper"
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
                _check(
                    request["values"]
                    == (
                        operator["arm_values"]
                        if name == "arm"
                        else [operator["normalized_closure"]]
                    ),
                    "interpreted operator request mismatch",
                )
                _check(
                    json.loads(request["observation"]["snapshot_json"])
                    == requested["observation_before"][
                        "arm" if name == "arm" else "gripper"
                    ],
                    "request observation binding",
                )
                values.extend(request["values"])
            _check(
                torch.equal(
                    torch.tensor(values, dtype=torch.float32), payload["actions"][i, 0]
                ),
                "submitted tensor differs from request",
            )
            _validate_observation(
                requested["observation_before"],
                payload["curr_obs"],
                i,
                i,
                logical_ns,
                manifest,
            )
            _validate_observation(
                outcome["observation_after"],
                payload["next_obs"],
                i,
                i + 1,
                logical_ns + manifest["step_period_ns"],
                manifest,
            )
            if i:
                for key in payload["curr_obs"]:
                    _check(
                        torch.equal(
                            payload["curr_obs"][key][i], payload["next_obs"][key][i - 1]
                        ),
                        "discontinuous observation chain",
                    )
        total += count
    _check(metadata["total_samples"] == total, "native sample count mismatch")
    buffer = TrajectoryReplayBuffer(sample_window_size=len(episodes))
    try:
        buffer.load_checkpoint(str(root / "demos"))
        stats = buffer.get_stats()
        _check(
            stats["num_trajectories"] == len(episodes)
            and stats["total_samples"] == total,
            "native reload counts",
        )
    finally:
        buffer.close()
    return {"episodes": len(episodes), "steps": total, "native_reload": True, **GATE}


def validate_synthetic_dataset(root: Path | str) -> dict:
    """Verify finalized owned artifacts and reload them through the native buffer."""
    try:
        return _validate_dataset(Path(root), "COMPLETE")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"Invalid synthetic dataset schema: {exc}") from exc
