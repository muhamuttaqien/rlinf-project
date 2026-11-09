# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Build an EXPERIMENTAL offline LeRobot dataset from FR3 motion proxies.

The actions are observed next-state TCP deltas and next measured gripper angle,
NOT controller-accepted commands. No physical execution approval is implied.
Never pass this output to the verified-controller exporter or a live robot.
Original raw episodes and proxy JSONs are opened read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.realworld_check.fr3_observed_action_proxy import (
    FIELDS, SCHEMA as PROXY_SCHEMA, analyze_episode,
)
from toolkits.realworld_check.fr3_policy_contract import (
    CONTRACT_VERSION, candidate_19d_state,
)

EXPORT_SCHEMA = "fr3_experimental_observed_proxy_lerobot_v1"


def _json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def validate_episode(raw: Path, proxy_file: Path) -> dict[str, Any]:
    """Check operator success, proxy provenance, and every action transition."""
    manifest_path = raw / "manifest.json"
    manifest = _json(manifest_path)
    review = _json(raw / "episode_review.json")
    if (review.get("operator_outcome") != "success"
            or review.get("capture_status") != "raw_recorded"):
        raise ValueError(f"{raw.name}: episode is not a reviewed completed success")

    summary, expected = analyze_episode(raw)
    proxy = _json(proxy_file)
    prior = proxy.get("summary")
    if proxy.get("schema") != PROXY_SCHEMA or not isinstance(prior, dict):
        raise ValueError(f"{raw.name}: wrong proxy format")
    if prior.get("raw_manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError(f"{raw.name}: stale/mismatched raw manifest")
    if (prior.get("action_source") != summary["action_source"]
            or prior.get("raw_episode") != raw.name
            or prior.get("action_fields") != list(FIELDS)):
        raise ValueError(f"{raw.name}: unexpected action provenance")
    if (summary["rejected_time_intervals"] != 0
            or len(expected) != summary["frames"] - 1):
        raise ValueError(f"{raw.name}: missing or rejected observation transitions")
    records = proxy.get("records")
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError(f"{raw.name}: proxy transition count mismatch")

    for row, correct in zip(records, expected, strict=True):
        if any(row.get(key) != correct[key] for key in (
                "frame_index", "next_frame_index",
                "observation_monotonic_ns", "next_observation_monotonic_ns")):
            raise ValueError(f"{raw.name}: misaligned proxy transition")
        action = np.asarray(row.get("observed_transition_proxy_7d"), dtype=np.float64)
        target = np.asarray(correct["observed_transition_proxy_7d"], dtype=np.float64)
        if (action.shape != (7,) or not np.isfinite(action).all()
                or not np.allclose(action, target, rtol=1e-8, atol=1e-10)):
            raise ValueError(f"{raw.name}: proxy action does not match raw states")
        if not np.isclose(row.get("dt_s", -1), correct["dt_s"], atol=1e-9):
            raise ValueError(f"{raw.name}: proxy time interval mismatch")

    return {
        "episode": raw.name,
        "frames": summary["frames"],
        "transitions": len(expected),
        "manifest_sha256": prior["raw_manifest_sha256"],
        "proxy_sha256": hashlib.sha256(proxy_file.read_bytes()).hexdigest(),
        "task": manifest["task"],
        "camera_profiles": manifest["camera_profiles"],
        "fps": manifest["sample_rate_hz_requested"],
    }


def validate_all(raw_root: Path, proxy_root: Path, expected_episodes: int) -> list[dict[str, Any]]:
    raw_root = raw_root.resolve()
    proxy_root = proxy_root.resolve()
    episodes = sorted(p for p in raw_root.glob("episode_*") if p.is_dir())
    if len(episodes) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episode directories, got {len(episodes)}")
    reports = [
        validate_episode(raw, proxy_root / (raw.name + ".json"))
        for raw in episodes
    ]
    first = reports[0]
    for report in reports[1:]:
        if (report["task"] != first["task"]
                or report["camera_profiles"] != first["camera_profiles"]
                or report["fps"] != first["fps"]):
            raise ValueError(f"{report['episode']}: task/camera/FPS mismatch")
    return reports


def export_dataset(raw_root: Path, proxy_root: Path, output: Path, *,
                   repo_id: str, reports: list[dict[str, Any]]) -> None:
    """Export NEW LeRobot dataset. The output remains experimental/unapproved."""
    if output.exists():
        raise FileExistsError(f"Output already exists; will not overwrite: {output}")
    if output.resolve().is_relative_to(raw_root.resolve()):
        raise ValueError("Output must be outside the raw demonstration root")
    if not repo_id.strip() or "/" not in repo_id:
        raise ValueError("repo-id must be like local/fr3_peg_proxy_v1")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from rlinf.data.storage.lerobot.compat import add_frame_to_dataset

    profiles = reports[0]["camera_profiles"]
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
    kwargs = {
        "repo_id": repo_id, "root": output,
        "fps": reports[0]["fps"], "robot_type": "fr3_robotiq",
        "features": features,
        "image_writer_threads": 1, "image_writer_processes": 0,
    }
    if "use_videos" in inspect.signature(LeRobotDataset.create).parameters:
        kwargs["use_videos"] = False

    dataset = LeRobotDataset.create(**kwargs)
    try:
        for report in reports:
            raw = raw_root / report["episode"]
            proxy = _json(proxy_root / (report["episode"] + ".json"))
            with (raw / "states.jsonl").open(encoding="utf-8") as stream:
                states = [json.loads(line) for line in stream if line.strip()]
            for i, record in enumerate(proxy["records"]):
                source = states[i]
                images = {}
                for role, column in (("base", "image"),
                                     ("wrist", "extra_view_image")):
                    relative = source["image_files_bgr8"][role]
                    if (Path(relative).is_absolute() or ".." in Path(relative).parts
                            or not relative.startswith(f"frames/{role}/")):
                        raise ValueError(f"Unsafe image path in {raw.name}: {relative}")
                    bgr = np.load(raw / relative, allow_pickle=False)
                    expected_shape = features[column]["shape"]
                    if bgr.shape != expected_shape or bgr.dtype != np.uint8:
                        raise ValueError(f"Bad image dimensions in {raw.name}: {relative}")
                    images[column] = np.ascontiguousarray(bgr[:, :, ::-1])

                terminal = i == len(proxy["records"]) - 1
                add_frame_to_dataset(dataset, {
                    **images,
                    "state": candidate_19d_state(source["measured_state"]),
                    "actions": np.asarray(
                        record["observed_transition_proxy_7d"], dtype=np.float32),
                    "done": np.array([terminal], dtype=np.bool_),
                    "is_success": np.array([terminal], dtype=np.bool_),
                    "task": report["task"],
                })
            dataset.save_episode()
            print(f"Exported {report['episode']}", flush=True)
        writer = getattr(dataset, "image_writer", None)
        if writer is not None:
            writer.wait_until_done()
    finally:
        writer = getattr(dataset, "image_writer", None)
        if writer is not None:
            writer.stop()
            dataset.image_writer = None
        del dataset

    provenance = {
        "schema": EXPORT_SCHEMA,
        "source": "observed next-state motion proxies, NOT controller actions",
        "state_contract": CONTRACT_VERSION,
        "action_fields": list(FIELDS),
        "episodes": len(reports),
        "transitions": sum(r["transitions"] for r in reports),
        "episode_sources": reports,
        "last_observation_excluded": True,
        "future_state_used_to_derive_actions": True,
        "is_controller_accepted_action": False,
        "controller_frame_and_gripper_calibration_approved": False,
        "physical_policy_execution_approved": False,
        "sft_ready": False,
        "recap_ready": False,
        "notes": (
            "Exploratory offline behavior-cloning data only. Must validate action "
            "semantics, policy observation mapping, normalization and execution "
            "interface before any physical deployment."
        ),
    }
    (output / "fr3_experimental_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-id", default="local/fr3_peg_proxy_v1")
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.expected_episodes < 1:
        parser.error("--expected-episodes must be positive")
    if not args.validate_only and args.output is None:
        parser.error("--output required unless --validate-only")
    reports = validate_all(args.raw_root, args.proxy_root, args.expected_episodes)
    total = sum(r["transitions"] for r in reports)
    print(f"Validated: {len(reports)} episodes, {total} observed-action proxies")
    if args.validate_only:
        print("No dataset written. Training/deployment approval: NO.")
        return 0
    export_dataset(
        args.raw_root.expanduser().resolve(),
        args.proxy_root.expanduser().resolve(),
        args.output.expanduser(),
        repo_id=args.repo_id, reports=reports,
    )
    print(f"Experimental LeRobot written: {args.output}")
    print("This dataset is NOT a verified controller-action dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
