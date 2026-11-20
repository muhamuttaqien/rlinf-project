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

"""Native-writer export and independent field audit of synthetic FR3 episodes."""

import importlib.metadata
import shutil
from pathlib import Path

import numpy as np
import torch

from rlinf.data.storage.fr3_synthetic import GATE, file_digest, read_json, write_json
from rlinf.data.storage.fr3_validation import (
    artifact,
    assess_synthetic_dataset,
    load_records,
    native_path,
    require,
)
from rlinf.data.storage.lerobot.compat import episode_boundaries
from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

EXPORT_SCHEMA = "fr3_synthetic_lerobot_export_v1"
TASK_LABEL = "synthetic FR3 pipeline validation (unscored; no physical execution)"
REPO_ID = "rlinf/fr3-synthetic-pipeline"


def _frames(root: Path, manifest: dict, index: dict, number: int) -> list[dict]:
    entry = manifest["episodes"][number]
    episode = read_json(artifact(root, entry["metadata_path"]))
    records = load_records(artifact(root, entry["provenance_path"]))
    payload = torch.load(
        native_path(root, index, number), map_location="cpu", weights_only=True
    )
    frames = []
    for i in range(episode["step_count"]):
        request, outcome = records[2 * i : 2 * i + 2]
        frame = {
            "task": TASK_LABEL,
            "state": payload["curr_obs"]["states"][i, 0].numpy(),
            "next_state": payload["next_obs"]["states"][i, 0].numpy(),
            "actions": payload["actions"][i, 0].numpy(),
        }
        for key in ("rewards", "dones", "terminations", "truncations"):
            frame[key] = payload[key][i, 0].numpy()
        for obs_key, prefix in (("curr_obs", ""), ("next_obs", "next_")):
            frame[prefix + "image"] = payload[obs_key]["main_images"][i, 0].numpy()
            frame[prefix + "extra_view_image"] = payload[obs_key]["extra_view_images"][
                i, 0, 0
            ].numpy()
        for key, value in (
            ("logical_ns", request["logical_action_ns"]),
            (
                "next_logical_ns",
                episode["start_logical_ns"] + (i + 1) * manifest["step_period_ns"],
            ),
            ("request_recorder_ns", request["recorder_ns"]),
            ("outcome_recorder_ns", outcome["recorder_ns"]),
        ):
            frame[key] = np.array([value], dtype=np.int64)
        frame["synthetic"] = np.array([True], dtype=bool)
        frame["physical_execution"] = np.array([False], dtype=bool)
        frames.append(frame)
    return frames


def _features(frame: dict) -> dict:
    features = {}
    for key, value in frame.items():
        if key == "task":
            continue
        is_image = key in {
            "image",
            "extra_view_image",
            "next_image",
            "next_extra_view_image",
        }
        features[key] = {
            "dtype": "image" if is_image else str(value.dtype),
            "shape": value.shape,
            "names": ["height", "width", "channel"] if is_image else None,
        }
    return features


def export_synthetic_dataset(
    root: Path | str, destination: Path | str, *, task_label: str = TASK_LABEL
) -> dict:
    """Export a validated dataset to a new directory, preserving exact native artifacts.

    The fixed task label describes synthetic pipeline verification. It is derived
    export metadata, not a recovered demonstration instruction or success label.
    Failed exports retain their partial artifacts and an explicit failure report.
    """
    root, destination = Path(root).resolve(), Path(destination).resolve()
    require(
        task_label == TASK_LABEL, "Expected the fixed unscored synthetic task label"
    )
    require(
        root != destination
        and root not in destination.parents
        and destination not in root.parents,
        "Export destination must be separate from the source dataset",
    )
    quality = assess_synthetic_dataset(root)
    require(
        quality["valid"],
        "Export requires a valid native synthetic dataset; inspect quality diagnostics",
    )
    manifest = read_json(root / "manifest.json")
    index = read_json(root / "demos/trajectory_index.json")
    period = manifest["step_period_ns"]
    require(
        1_000_000_000 % period == 0,
        "Native writer requires an integer FPS; no resampling is authorized",
    )
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": EXPORT_SCHEMA,
        "status": "WRITING",
        **GATE,
        "source_manifest_sha256": quality["source_manifest_sha256"],
        "task": TASK_LABEL,
        "task_origin": "derived synthetic pipeline label; original instruction unavailable",
        "native_archive": "native",
        "dataset_root": "lerobot",
        "repo_id": REPO_ID,
        "field_mapping": {
            "state": "curr_obs.states[step,0]",
            "next_state": "next_obs.states[step,0]",
            "actions": "submitted normalized mock payload, never physical execution",
            "image": "D435i current frame",
            "extra_view_image": "D405 current frame",
            "timestamp": "episode-relative frame_index / synthetic FPS",
        },
        "sidecar_only_fields": [
            "typed_requests",
            "action_events",
            "operator_request",
            "stream_source_and_receipt_timing",
            "clock_domains",
            "session_and_authority",
            "unavailable_measurements",
            "configuration_and_interface_hashes",
        ],
        "unsupported": [
            "real task instructions",
            "physical calibration",
            "physical controller acceptance",
            "physical outcomes",
        ],
        "files": {},
    }
    writer = LeRobotDatasetWriter()
    try:
        report["lerobot_version"] = importlib.metadata.version("lerobot")
        archive = destination / "native"
        for relative in ["manifest.json", *manifest["files"]]:
            source = artifact(root, relative)
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        frames = _frames(root, manifest, index, 0)
        writer.create(
            repo_id=REPO_ID,
            root=destination / "lerobot",
            robot_type="fr3_synthetic",
            fps=1_000_000_000 // period,
            features=_features(frames[0]),
            image_writer_threads=0,
            image_writer_processes=0,
        )
        writer.add_episode(frames)
        for number in range(1, len(manifest["episodes"])):
            writer.add_episode(_frames(root, manifest, index, number))
        writer.finalize()
        report["files"] = {
            str(p.relative_to(destination)): file_digest(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        }
        report["status"] = "VERIFYING"
        write_json(destination / "export.json", report)
        report["audit"] = audit_synthetic_export(
            destination, expected_status="VERIFYING"
        )
        report["status"] = "COMPLETE"
        write_json(destination / "export.json", report)
        return report
    except Exception as exc:
        report.update(
            status="FAILED", error={"type": type(exc).__name__, "message": str(exc)}
        )
        write_json(destination / "export.json", report)
        raise
    finally:
        if writer.dataset is not None:
            writer.finalize()


def audit_synthetic_export(
    destination: Path | str, *, expected_status: str = "COMPLETE"
) -> dict:
    """Reload through LeRobot and compare every exported field with archived native data."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    destination = Path(destination).absolute()
    report = read_json(artifact(destination, "export.json"))
    require(
        report["schema"] == EXPORT_SCHEMA and report["status"] == expected_status,
        "Incomplete/unknown export schema",
    )
    require(
        all(report.get(k) == v for k, v in GATE.items()),
        "Invalid export execution claim",
    )
    require(report["task"] == TASK_LABEL, "Invalid export task label")
    actual = {
        str(p.relative_to(destination))
        for p in destination.rglob("*")
        if p.is_file() and p.name != "export.json"
    }
    require(actual == set(report["files"]), "Missing/unindexed export artifact")
    for relative, digest in report["files"].items():
        require(
            file_digest(artifact(destination, relative)) == digest,
            f"Corrupt export artifact: {relative}",
        )
    root = destination / "native"
    quality = assess_synthetic_dataset(root)
    require(
        quality["valid"]
        and quality["source_manifest_sha256"] == report["source_manifest_sha256"],
        "Native provenance archive failed validation",
    )
    manifest, index = (
        read_json(root / "manifest.json"),
        read_json(root / "demos/trajectory_index.json"),
    )
    dataset = LeRobotDataset(
        repo_id=REPO_ID, root=destination / "lerobot", download_videos=False
    )
    starts, ends = episode_boundaries(dataset)
    require(
        len(starts) == len(manifest["episodes"]) == dataset.num_episodes,
        "Exported episode count mismatch",
    )
    require(
        len(dataset) == quality["counts"]["transitions"],
        "Exported transition count mismatch",
    )
    compared = 0
    field_counts = {}
    for number, (start, end) in enumerate(zip(starts, ends)):
        frames = _frames(root, manifest, index, number)
        require(end - start == len(frames), "Export episode boundary mismatch")
        for step, expected in enumerate(frames):
            actual = dataset[start + step]
            require(
                int(actual["episode_index"]) == number
                and int(actual["frame_index"]) == step
                and int(actual["index"]) == start + step,
                "Incorrect export frame/episode/index association",
            )
            require(
                abs(
                    float(actual["timestamp"]) - step * manifest["step_period_ns"] / 1e9
                )
                < 1e-7,
                "Incorrect export relative timestamp",
            )
            for key, value in expected.items():
                if key == "task":
                    require(actual["task"] == value, "Task metadata mismatch")
                else:
                    loaded = actual[key].numpy()
                    # LeRobot's HF transform squeezes one-element numeric features.
                    if value.shape == (1,) and loaded.shape == ():
                        loaded = loaded.reshape(1)
                    if value.ndim == 3:
                        require(
                            loaded.shape == (3, *value.shape[:2])
                            and np.isfinite(loaded).all()
                            and loaded.min() >= 0
                            and loaded.max() <= 1,
                            "Invalid decoded image layout/range",
                        )
                        loaded = np.rint(loaded.transpose(1, 2, 0) * 255).astype(
                            np.uint8
                        )
                    require(
                        loaded.dtype == value.dtype and np.array_equal(loaded, value),
                        f"Export field differs from native: episode={number} step={step} field={key}",
                    )
                field_counts[key] = field_counts.get(key, 0) + 1
            compared += 1
    return {
        "status": "PASS",
        "episodes": len(starts),
        "transitions": compared,
        "fields_compared": field_counts,
        "native_archive_hash_verified": True,
        "independent_loader": "LeRobotDataset",
        **GATE,
    }
