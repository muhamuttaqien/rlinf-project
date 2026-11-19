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

"""Audit saved FR3 streams and export a separate causal 10 Hz request dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

from rlinf.utils.logging import get_logger
from toolkits.realworld_check.fr3_command_request_alignment import (
    _grip,
    _mean_held_twist,
    _twist,
)
from toolkits.realworld_check.fr3_policy_contract import candidate_19d_state
from toolkits.realworld_check.fr3_request_norm_stats import compute_from_arrays

LOGGER = get_logger()
SCHEMA = "fr3_frequency_consistent_request_v1"
PERIOD_NS = 100_000_000
STATE_KEYS = ("joint_state", "tcp_pose", "external_wrench", "gripper_state")
ROLES = ("base", "wrist")


def read_json(path: Path) -> dict:
    """Read one JSON object from an existing file."""
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Expected an object in {path}")
    return result


def read_rows(path: Path) -> list[dict]:
    """Read saved JSONL records without touching any hardware interface."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    """Hash source metadata for before/after integrity checks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list | np.ndarray) -> dict:
    """Summarize finite observed intervals, excluding inter-episode boundaries."""
    data = np.asarray(values, dtype=np.float64)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Expected nonempty finite timing measurements")
    return dict(
        zip(
            ("min", "p50", "p95", "p99", "max"),
            np.quantile(data, [0, 0.5, 0.95, 0.99, 1]).tolist(),
            strict=True,
        )
    )


def receipt_times(row: dict) -> dict[str, int]:
    """Return all six measurements' workstation monotonic receipt times."""
    meta = row["source_and_camera_metadata"]
    result = {key: meta["source"][key]["received_monotonic_ns"] for key in STATE_KEYS}
    result.update(
        {role: meta["cameras"][role]["received_monotonic_ns"] for role in ROLES}
    )
    if any(type(value) is not int or value <= 0 for value in result.values()):
        raise ValueError("All modalities require positive host-monotonic timestamps")
    return result


def fixed_grid(available: list[int], stop: int) -> np.ndarray:
    """Fit a causal 100 ms lattice while retaining each saved observation.

    The smallest phase satisfying t[i] >= available[i] preserves the measured
    tuple. Reject clock drift large enough to skip or duplicate source frames.
    """
    stamps = np.asarray(available, dtype=np.int64)
    if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
        raise ValueError("Observation availability must strictly increase")
    phases = stamps - np.arange(len(stamps), dtype=np.int64) * PERIOD_NS
    if int(np.ptp(phases)) >= PERIOD_NS:
        raise ValueError("Clock drift/gaps exceed one 100 ms period")
    anchor = int(phases.max())
    count = (min(stop, int(stamps[-1]) + PERIOD_NS) - anchor) // PERIOD_NS
    grid = anchor + np.arange(count, dtype=np.int64) * PERIOD_NS
    if count < 10:
        raise ValueError("Episode has fewer than ten complete 100 ms intervals")
    indices = np.searchsorted(stamps, grid, side="right") - 1
    if not np.array_equal(indices, np.arange(count)):
        raise ValueError("The fixed grid would skip or repeat original observations")
    return grid


def held_integral(
    start: int, stop: int, stamps: list[int], values: list[list]
) -> np.ndarray:
    """Independently integrate a held normalized request over a retained span."""
    cuts = np.asarray(
        [start, *[s for s in stamps if start < s < stop], stop], dtype=np.int64
    )
    indices = np.searchsorted(stamps, cuts[:-1], side="right") - 1
    if np.any(indices < 0):
        raise ValueError("Cannot integrate before the first recorded request")
    return (
        np.asarray(values, dtype=np.float64)[indices]
        * (np.diff(cuts).astype(np.float64) / 1e9)[:, None]
    ).sum(axis=0)


def split_assignments(train: Path, validation: Path) -> dict[str, str]:
    """Reuse exact existing episode identities; never randomize a new split."""
    result = {}
    for split, path in (("train", train), ("validation", validation)):
        provenance = read_json(path / "fr3_command_request_provenance.json")
        if provenance["source_split"] != split:
            raise ValueError(
                "Reference dataset split identity disagrees with its provenance"
            )
        names = {segment["episode"] for segment in provenance["segments"]}
        if len(names) != provenance["source_episodes"]:
            raise ValueError("Reference split has inconsistent source episodes")
        for name in names:
            if name in result:
                raise ValueError(f"Train/validation leakage: {name}")
            result[name] = split
    return result


def process_episode(ep: Path, split: str) -> tuple[dict, list[dict], dict]:
    """Audit one complete episode and build fully covered causal request bins."""
    manifest = read_json(ep / "manifest.json")
    review = read_json(ep / "episode_review.json")
    if (
        manifest["schema"] != "fr3_robotiq_raw_capture_v1"
        or manifest["status"] != "raw_recorded"
        or manifest["operator_events_dropped"] != 0
        or manifest["failures"]
        or review["operator_outcome"] != "success"
        or not review["operator_reviewed"]
    ):
        raise ValueError(
            f"{ep.name}: incomplete capture or missing genuine outcome review"
        )
    rows, events = (
        read_rows(ep / "states.jsonl"),
        read_rows(ep / "operator_commands.jsonl"),
    )
    if (
        len(rows) != manifest["frames_saved"]
        or len(events) != manifest["operator_command_events_saved"]
    ):
        raise ValueError(f"{ep.name}: source counts disagree")
    receipts = [receipt_times(row) for row in rows]
    streams = {key: [r[key] for r in receipts] for key in (*STATE_KEYS, *ROLES)}
    streams["recorded"] = [r["recorded_monotonic_ns"] for r in rows]
    twists, grips = [], []
    for event in events:
        stamp = event["received_monotonic_ns"]
        if event["name"] == "operator_twist":
            if event.get("units") != {
                "linear": "normalized_controller_input_not_m_per_s",
                "angular": "normalized_controller_input_not_rad_per_s",
            }:
                raise ValueError(
                    "Expected normalized held requests, not metric velocity/delta labels"
                )
            twists.append((stamp, _twist(event)))
        elif event["name"] == "operator_gripper_request":
            grips.append((stamp, _grip(event)))
        else:
            raise ValueError("Unexpected raw request type")
    if not twists or not grips:
        raise ValueError("Missing arm or gripper request stream")
    ts, values = [x[0] for x in twists], [x[1] for x in twists]
    gs, gv = [x[0] for x in grips], [x[1] for x in grips]
    streams.update(operator_twist=ts, operator_gripper_request=gs)
    for key, stamps in streams.items():
        if np.any(np.diff(stamps) <= 0):
            raise ValueError(f"{ep.name}: duplicate/reordered timestamps in {key}")
    if max(np.diff(ts)) > 100_000_000:
        raise ValueError(f"{ep.name}: stale arm-request gap")
    available = [row["recorded_monotonic_ns"] for row in rows]
    if any(
        max(receipt.values()) > stamp
        for receipt, stamp in zip(receipts, available, strict=True)
    ):
        raise ValueError("A saved observation cannot precede its source receipts")
    stop = review["capture_end_workstation_monotonic_ns"]
    grid = fixed_grid(available, stop)
    output = []
    ages, skews, record_lags, old_future = [], [], [], []
    camera_skews, state_skews = [], []
    header_deltas = defaultdict(list)
    for i, row in enumerate(rows):
        if row["frame_index"] != i or row["training_action"] is not None:
            raise ValueError("Original frame indices or raw action fields were altered")
        meta = row["source_and_camera_metadata"]
        if meta["arm_joint_names"] != [f"fr3_joint{j}" for j in range(1, 8)]:
            raise ValueError("Unexpected arm joint ordering")
        if meta["gripper_joint_name"] != "robotiq_85_left_knuckle_joint":
            raise ValueError(
                "Expected physical knuckle measurement, not normalized gripper state"
            )
        if meta["gripper_units"] != "knuckle_radians_not_metric_width":
            raise ValueError("Gripper state must remain physical knuckle radians")
        if any(
            meta["source"][key]["header"]["frame_id"] != "base"
            for key in ("tcp_pose", "external_wrench")
        ):
            raise ValueError("Pose and wrench must preserve the recorded base frame")
        state = candidate_19d_state(row["measured_state"])
        for role in ROLES:
            rel = row["image_files_bgr8"][role]
            if (
                Path(rel).is_absolute()
                or ".." in Path(rel).parts
                or not rel.startswith(f"frames/{role}/")
            ):
                raise ValueError("Unsafe camera frame reference")
            image = np.load(ep / rel, mmap_mode="r", allow_pickle=False)
            w, h = manifest["camera_profiles"][role]["resolution"]
            if image.shape != (h, w, 3) or image.dtype != np.uint8:
                raise ValueError("Camera frame shape/dtype mismatch")
        stamps = receipts[i]
        skews.append((max(stamps.values()) - min(stamps.values())) / 1e9)
        camera_skews.append(abs(stamps["base"] - stamps["wrist"]) / 1e9)
        state_skews.append(
            (max(stamps[k] for k in STATE_KEYS) - min(stamps[k] for k in STATE_KEYS))
            / 1e9
        )
        for key in STATE_KEYS:
            source = meta["source"][key]
            header = source["header"]["stamp"]
            header_deltas[key].append(
                (
                    source["received_unix_ns"]
                    - (header["sec"] * 1_000_000_000 + header["nanosec"])
                )
                / 1e9
            )
        record_lags.append((row["recorded_monotonic_ns"] - max(stamps.values())) / 1e9)
        old_future.append(
            (max(stamps[k] for k in STATE_KEYS) - max(stamps[k] for k in ROLES)) / 1e9
        )
        if i >= len(grid):
            continue
        start, end = int(grid[i]), int(grid[i]) + PERIOD_NS
        age = (start - min(stamps.values())) / 1e9
        if max(stamps.values()) > start or not 0 <= age <= 0.1:
            raise ValueError(
                f"{ep.name}: noncausal or stale observation at grid index {i}"
            )
        ages.append(age)
        grip_index = bisect_right(gs, start) - 1
        if grip_index < 0:
            continue  # Preserve unknown initial button state, never infer it from knuckle angle.
        mean, coverage = _mean_held_twist(start, end, ts, values, 100_000_000)
        if mean is None or abs(coverage - 1.0) > 1e-12:
            raise ValueError(f"{ep.name}: incomplete arm request interval")
        output.append(
            {
                "episode": ep.name,
                "source_frame_index": i,
                "source_recorded_ns": row["recorded_monotonic_ns"],
                "grid_start_ns": start,
                "grid_end_ns": end,
                "source_receipts_ns": stamps,
                "state": state.tolist(),
                "actions": [*mean, gv[grip_index]],
                "images_bgr8": row["image_files_bgr8"],
            }
        )
    if len(output) < 10:
        raise ValueError("Too few known-gripper samples for one complete horizon")
    grid_starts = [r["grid_start_ns"] for r in output]
    mapped_events = []
    for stamp, value in grips:
        index = bisect_left(grid_starts, stamp)
        if index == len(output) or output[index]["actions"][6] != value:
            raise ValueError("Gripper event would be lost or merged on the 10 Hz grid")
        delay = (grid_starts[index] - stamp) / 1e9
        if not 0 <= delay < 0.1:
            raise ValueError("Gripper event delayed by more than one grid period")
        mapped_events.append(
            {
                "source_ns": stamp,
                "request": value,
                "grid_ns": grid_starts[index],
                "delay_s": delay,
            }
        )
    expected = held_integral(
        output[0]["grid_start_ns"], output[-1]["grid_end_ns"], ts, values
    )
    actions = np.asarray([r["actions"] for r in output])
    error = float(np.abs(actions[:, :6].sum(axis=0) * 0.1 - expected).max())
    if error > 1e-10:
        raise ValueError("Resampling did not preserve the normalized request integral")
    indices = [r["source_frame_index"] for r in output]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Retained trajectory skips or repeats source observations")
    source_pose = np.asarray(
        [rows[i]["measured_state"]["tcp_pose"][:3] for i in indices]
    )
    output_pose = np.asarray([r["state"][:3] for r in output])
    pose_error = float(np.linalg.norm(source_pose - output_pose, axis=1).max())
    report = {
        "episode": ep.name,
        "split": split,
        "raw_samples": len(rows),
        "grid_intervals": len(grid),
        "retained_samples": len(output),
        "unknown_gripper_prefix_bins": indices[0],
        "first_source_frame": indices[0],
        "last_source_frame": indices[-1],
        "grid_anchor_ns": int(grid[0]),
        "last_grid_end_ns": output[-1]["grid_end_ns"],
        "retained_duration_s": len(output) * 0.1,
        "terminal_tail_trim_s": (stop - output[-1]["grid_end_ns"]) / 1e9,
        "observation_age_s": summary(ages),
        "tuple_skew_s": summary(skews),
        "gripper_events": mapped_events,
        "request_integral_error_float64": error,
        "request_integral_error_float32": float(
            np.abs(
                actions[:, :6].astype(np.float32).astype(np.float64).sum(axis=0) * 0.1
                - expected
            ).max()
        ),
        "retained_pose_error_m": pose_error,
        "retained_path_length_m": float(
            np.linalg.norm(np.diff(source_pose, axis=0), axis=1).sum()
        ),
        "source_sha256": {
            name: digest(ep / name)
            for name in (
                "manifest.json",
                "states.jsonl",
                "operator_commands.jsonl",
                "episode_review.json",
            )
        },
    }
    measurements = {
        "streams": streams,
        "tuple_skew_s": skews,
        "camera_receipt_skew_s": camera_skews,
        "state_receipt_skew_s": state_skews,
        "record_lag_s": record_lags,
        "old_state_after_label_start_s": old_future,
        **{
            f"host_unix_minus_header_{key}_s": values
            for key, values in header_deltas.items()
        },
    }
    return report, output, measurements


def audit(
    raw: Path, train: Path, validation: Path
) -> tuple[dict, dict[str, list[dict]]]:
    """Read every episode, preserving existing identities and stream clocks."""
    assignments = split_assignments(train, validation)
    episodes = sorted(p for p in raw.glob("episode_*") if p.is_dir())
    if {p.name for p in episodes} != set(assignments) or len(episodes) != 100:
        raise ValueError(
            "Expected the same 100 source episodes as the existing 80/20 split"
        )
    durations, rates, counts = defaultdict(list), defaultdict(list), defaultdict(int)
    sync = defaultdict(list)
    reports, converted = [], {}
    session = read_json(raw / "session.json")
    for ep in episodes:
        manifest = read_json(ep / "manifest.json")
        if (
            manifest["camera_profiles"] != session["camera_profiles"]
            or manifest["task"] != session["task"]
        ):
            raise ValueError("Camera profiles or task changed between source episodes")
        report, rows, measured = process_episode(ep, assignments[ep.name])
        reports.append(report)
        converted[ep.name] = rows
        for key, stamps in measured.pop("streams").items():
            durations[key].extend((np.diff(stamps) / 1e9).tolist())
            rates[key].append((len(stamps) - 1) / ((stamps[-1] - stamps[0]) / 1e9))
            counts[key] += len(stamps)
        for key, values in measured.items():
            sync[key].extend(values)
    result = {
        "schema": SCHEMA,
        "raw_root": str(raw.resolve()),
        "reference_splits": {
            "train": str(train.resolve()),
            "validation": str(validation.resolve()),
        },
        "reference_split_sha256": {
            "train": digest(train / "fr3_command_request_provenance.json"),
            "validation": digest(validation / "fr3_command_request_provenance.json"),
        },
        "target_hz": {
            "raw_requests": 100,
            "available_saved_state": 10,
            "dataset": 10,
            "policy_execution": 10,
            "command_refresh": 100,
        },
        "action_semantics": "six normalized held controller requests; not SI velocity, pose deltas, or executed motion",
        "resampling": "causal completed saved tuples; 100 ms time-weighted held-request means; gripper latch at interval START",
        "clock": "workstation monotonic receipt; camera exposure and controller acceptance timestamps unavailable",
        "streams": {
            key: {
                "samples": counts[key],
                "gap_s": summary(durations[key]),
                "episode_rate_hz": summary(rates[key]),
                "nonpositive_gaps": sum(x <= 0 for x in durations[key]),
            }
            for key in durations
        },
        "synchronization": {key: summary(values) for key, values in sync.items()},
        "old_labels_with_future_state": sum(
            x > 0 for x in sync["old_state_after_label_start_s"]
        ),
        "episodes": reports,
        "offline_conversion_valid": True,
        "training_started": False,
        "physical_execution_approved": False,
    }
    result["existing_labels"] = audit_existing_labels(train, validation)
    return result, converted


def audit_existing_labels(train: Path, validation: Path) -> dict:
    """Measure actual old label intervals independently of their declared FPS."""
    import pyarrow.parquet as pq

    intervals, timestamps, errors = [], [], []
    frames = 0
    for split in (train, validation):
        provenance = read_json(split / "fr3_command_request_provenance.json")
        candidate = Path(provenance["source_candidate_root"])
        for index, segment in enumerate(provenance["segments"]):
            rows = read_rows(candidate / "episodes" / (segment["episode"] + ".jsonl"))[
                segment["start"] : segment["stop"]
            ]
            table = pq.read_table(
                split / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet",
                columns=["timestamp", "actions"],
            )
            actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
            expected = np.asarray(
                [row["candidate_request_action_7d"] for row in rows], dtype=np.float32
            )
            if not np.array_equal(actions, expected):
                raise ValueError(
                    "Existing SFT labels disagree with their conversion provenance"
                )
            dt = np.asarray([row["observation_dt_s"] for row in rows])
            intervals.extend(dt.tolist())
            timestamps.extend(np.diff(table["timestamp"].to_numpy()).tolist())
            errors.append(
                float(
                    np.abs(
                        (actions[:, :6].astype(np.float64) * (0.1 - dt[:, None])).sum(
                            axis=0
                        )
                    ).max()
                )
            )
            frames += len(rows)
    return {
        "frames": frames,
        "source_label_interval_s": summary(intervals),
        "exported_timestamp_gap_s": summary(timestamps),
        "max_episode_request_integral_change_from_retiming_s": max(errors),
        "task_prompt": read_json(train / "meta" / "tasks.jsonl")["task"],
    }


def export_split(
    raw: Path, output: Path, split: str, reports: list[dict], rows: dict, threads: int
) -> dict:
    """Write new LeRobot data with embedded lossless RGB and integer source clocks."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.data.storage.lerobot.compat import add_frame_to_dataset

    selected = [r for r in reports if r["split"] == split]
    profile = read_json(raw / selected[0]["episode"] / "manifest.json")[
        "camera_profiles"
    ]
    features = {
        key: {
            "dtype": "image",
            "shape": (
                profile[role]["resolution"][1],
                profile[role]["resolution"][0],
                3,
            ),
            "names": ["height", "width", "channel"],
        }
        for role, key in (("base", "image"), ("wrist", "extra_view_image"))
    }
    features.update(
        {
            "state": {"dtype": "float32", "shape": (19,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
            "is_success": {"dtype": "bool", "shape": (1,), "names": ["is_success"]},
            "source_time_ns": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["workstation_monotonic_ns"],
            },
            "source_frame_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["raw_frame_index"],
            },
        }
    )
    dataset = LeRobotDataset.create(
        repo_id=f"local/fr3_peg_request_10hz_phase1_{split}",
        root=output,
        fps=10,
        robot_type="fr3_robotiq",
        features=features,
        use_videos=False,
        image_writer_threads=threads,
        image_writer_processes=0,
    )
    index = []
    try:
        for number, report in enumerate(selected):
            name = report["episode"]
            episode_rows = rows[name]
            for i, row in enumerate(episode_rows):
                images = {
                    key: np.ascontiguousarray(
                        np.load(
                            raw / name / row["images_bgr8"][role], allow_pickle=False
                        )[..., ::-1]
                    )
                    for role, key in (("base", "image"), ("wrist", "extra_view_image"))
                }
                terminal = i == len(episode_rows) - 1
                add_frame_to_dataset(
                    dataset,
                    {
                        **images,
                        "state": np.asarray(row["state"], dtype=np.float32),
                        "actions": np.asarray(row["actions"], dtype=np.float32),
                        "done": np.asarray([terminal], dtype=bool),
                        "is_success": np.asarray([terminal], dtype=bool),
                        "source_time_ns": np.asarray(
                            [row["grid_start_ns"]], dtype=np.int64
                        ),
                        "source_frame_index": np.asarray(
                            [row["source_frame_index"]], dtype=np.int64
                        ),
                        "task": read_json(raw / name / "manifest.json")["task"],
                    },
                )
            dataset.save_episode()
            index.append(
                {
                    "dataset_episode_index": number,
                    "source_episode": name,
                    "frames": len(episode_rows),
                }
            )
            LOGGER.info("Exported %s %s: %d frames", split, name, len(episode_rows))
    finally:
        dataset.stop_image_writer()
    result = {
        "schema": SCHEMA,
        "source_split": split,
        "episodes": index,
        "fps": 10,
        "action_semantics": "normalized requests; gripper is not a knuckle measurement",
        "training_started": False,
        "physical_execution_approved": False,
    }
    (output / "frequency_provenance.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def validate_split(raw: Path, output: Path, converted: dict) -> dict:
    """Independently read every exported label and RGB pixel back from Parquet."""
    import pyarrow.parquet as pq
    from PIL import Image

    provenance = read_json(output / "frequency_provenance.json")
    count = 0
    for episode in provenance["episodes"]:
        index, name = episode["dataset_episode_index"], episode["source_episode"]
        file = output / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
        cursor = 0
        for batch in pq.ParquetFile(file).iter_batches(batch_size=8):
            for saved in batch.to_pylist():
                reference = converted[name][cursor]
                for key in ("state", "actions"):
                    if not np.array_equal(
                        np.asarray(saved[key], dtype=np.float32),
                        np.asarray(reference[key], dtype=np.float32),
                    ):
                        raise ValueError(f"{name}: {key} changed during export")
                if abs(saved["timestamp"] - cursor / 10) > 2e-6 or np.atleast_1d(
                    saved["source_time_ns"]
                ).tolist() != [reference["grid_start_ns"]]:
                    raise ValueError("Incorrect uniform dataset timestamp/source clock")
                if np.atleast_1d(saved["source_frame_index"]).tolist() != [
                    reference["source_frame_index"]
                ]:
                    raise ValueError("Source frame identity mismatch")
                for role, key in (("base", "image"), ("wrist", "extra_view_image")):
                    image = np.asarray(
                        Image.open(io.BytesIO(saved[key]["bytes"])).convert("RGB")
                    )
                    source = np.load(
                        raw / name / reference["images_bgr8"][role],
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    if not np.array_equal(image, source[..., ::-1]):
                        raise ValueError(
                            "Lossless RGB roundtrip differs from original BGR frame"
                        )
                if bool(np.atleast_1d(saved["done"])[0]) != (
                    cursor == episode["frames"] - 1
                ):
                    raise ValueError("Incorrect episode boundary")
                cursor += 1
        if cursor != episode["frames"]:
            raise ValueError("Exported episode length mismatch")
        count += cursor
    return {
        "episodes": len(provenance["episodes"]),
        "frames": count,
        "all_rgb_pixels_equal": True,
        "all_labels_equal": True,
        "timestamps_hz": 10,
        "episode_boundaries_valid": True,
    }


def main() -> None:
    """Audit before export; write only a new directory outside all source roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=Path("/data1/Muttaqien/fr3_peg_100")
    )
    parser.add_argument(
        "--reference-train",
        type=Path,
        default=Path("/data1/Muttaqien/fr3_peg_request_lerobot_train_v1"),
    )
    parser.add_argument(
        "--reference-validation",
        type=Path,
        default=Path("/data1/Muttaqien/fr3_peg_request_lerobot_validation_v1"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--image-threads", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    sources = [
        p.resolve()
        for p in (args.raw_root, args.reference_train, args.reference_validation)
    ]
    if output.exists() or any(
        output == p or output.is_relative_to(p) or p.is_relative_to(output)
        for p in sources
    ):
        raise ValueError("Output must be NEW and disjoint from every original dataset")
    report, rows = audit(args.raw_root, args.reference_train, args.reference_validation)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    aligned = output / "resampled"
    aligned.mkdir()
    for name, episode in rows.items():
        (aligned / (name + ".jsonl")).write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in episode)
        )
    if args.audit_only:
        LOGGER.info("Read-only source audit completed: %s", output)
        return
    report["export_validation"] = {}
    for split in ("train", "validation"):
        export_split(
            args.raw_root,
            output / split,
            split,
            report["episodes"],
            rows,
            args.image_threads,
        )
        report["export_validation"][split] = validate_split(
            args.raw_root, output / split, rows
        )
    train_names = {r["episode"] for r in report["episodes"] if r["split"] == "train"}
    train_rows = [
        r for name, episode in rows.items() if name in train_names for r in episode
    ]
    stats = compute_from_arrays(
        np.asarray([r["state"] for r in train_rows], dtype=np.float32),
        np.asarray([r["actions"] for r in train_rows], dtype=np.float32),
    )
    norm = output / "norm"
    norm.mkdir()
    (norm / "norm_stats.json").write_text(
        json.dumps({"norm_stats": stats}, indent=2) + "\n"
    )
    (norm / "provenance.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "split": "train_only",
                "training_samples": len(train_rows),
                "validation_samples_used": 0,
                "training_episodes": sorted(train_names),
            },
            indent=2,
        )
        + "\n"
    )
    for ep in report["episodes"]:
        for name, value in ep["source_sha256"].items():
            if digest(args.raw_root / ep["episode"] / name) != value:
                raise ValueError("Original recording metadata changed")
    report["original_metadata_unchanged"] = True
    report["complete"] = True
    (output / "audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    LOGGER.info("Validated separate 10 Hz dataset: %s", output)


if __name__ == "__main__":
    main()
