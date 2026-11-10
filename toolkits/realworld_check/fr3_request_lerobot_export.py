# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""OFFLINE LeRobot export of normalized FR3 SpaceMouse REQUEST candidates.

NO verified controller actions. NO controller integration, robot execution,
training, simulation success, or RECAP approval. Never load this export into a
proxy-action model/configuration: its 7D values have different units/semantics.
Input is the immutable candidate alignment + quality split report; missing
gripper prefixes and discontinuous action spans are explicitly omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np

from toolkits.realworld_check.fr3_command_request_alignment import FIELDS
from toolkits.realworld_check.fr3_command_request_quality import SCHEMA as QUALITY_SCHEMA

EXPORT_SCHEMA = "fr3_experimental_spacemouse_request_lerobot_v1"


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required at {path}")
    return data


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _is_vec(v: object, n: int) -> bool:
    return (isinstance(v, list) and len(v) == n
            and all(type(x) in (int, float) and math.isfinite(x) for x in v))


def preflight(candidate: Path, quality: Path, split: str,
              *, min_horizon: int = 10) -> dict:
    """Validate source/report relationship and return noncrossing source spans.

    Opening these sources is read-only. No LeRobot imports or writes here.
    """
    if split not in ("train", "validation"):
        raise ValueError("Expected split=train or validation")
    candidate = candidate.expanduser().resolve()
    quality = quality.expanduser().resolve()
    q = _json(quality)
    a = _json(candidate / "report.json")
    if q.get("schema") != QUALITY_SCHEMA or a.get("schema") != \
            "fr3_spacemouse_request_alignment_candidate_v1":
        raise ValueError("Wrong candidate or audit schema")
    if Path(q.get("source_candidate_root", "")).resolve() != candidate:
        raise ValueError("Quality manifest references a different candidate dataset")
    if q.get("horizon") != min_horizon:
        raise ValueError("Requested horizon does not match audited horizon")
    if (q.get("gripper_unknown_imputed") is not False
            or q.get("sft_ready") is not False
            or a.get("sft_ready") is not False
            or a.get("lerobot_exported") is not False):
        raise ValueError("Unsafe or mislabeled source provenance")
    raw = Path(a["source_root"]).resolve()
    sources = {x["episode"]: x for x in a["episode_reports"]}
    manifests = q.get("episode_manifests")
    if not isinstance(manifests, list) or set(x["episode"] for x in manifests) != set(sources):
        raise ValueError("Quality/source episode sets differ")
    selected = [x for x in manifests if x.get("split") == split]
    if not selected:
        raise ValueError("Split contains zero episodes")
    total = 0
    segments = []
    task = None
    profiles = None
    fps = None
    for info in selected:
        name = info["episode"]
        if not name.startswith("episode_") or "/" in name or ".." in name:
            raise ValueError("Unsafe episode name")
        ep = (raw / name).resolve()
        if ep.parent != raw:
            raise ValueError("Unsafe raw episode path")
        m = _json(ep / "manifest.json")
        review = _json(ep / "episode_review.json")
        if (review.get("operator_outcome") != "success"
                or review.get("capture_status") != "raw_recorded"):
            raise ValueError(f"{name}: success was not operator-reviewed")
        digest = hashlib.sha256((ep / "manifest.json").read_bytes()).hexdigest()
        if digest != sources[name].get("raw_manifest_sha256"):
            raise ValueError(f"{name}: original manifest changed")
        commands_digest = hashlib.sha256(
            (ep / "operator_commands.jsonl").read_bytes()
        ).hexdigest()
        if commands_digest != sources[name].get("recorded_requests_sha256"):
            raise ValueError(f"{name}: operator requests changed")
        if task is None:
            task, profiles, fps = (
                m.get("task"), m.get("camera_profiles"),
                m.get("sample_rate_hz_requested"),
            )
        if (m.get("task"), m.get("camera_profiles"),
                m.get("sample_rate_hz_requested")) != (task, profiles, fps):
            raise ValueError("Task, camera or requested FPS inconsistent across episodes")
        rows = _jsonl(candidate / "episodes" / (name + ".jsonl"))
        if len(rows) != info["transitions"]:
            raise ValueError(f"{name}: transition count mismatch")
        local_count = 0
        for start, stop in info["valid_spans_end_exclusive"]:
            if not (0 <= start < stop <= len(rows)):
                raise ValueError(f"{name}: bad segment limits")
            if stop - start < min_horizon:
                continue  # No padded action windows or synthetic sample indices.
            for idx in range(start, stop):
                row = rows[idx]
                act = row.get("candidate_request_action_7d")
                state = row.get("state19_candidate")
                if (row.get("frame_index") != idx
                        or row.get("candidate_complete") is not True
                        or not _is_vec(act, 7) or not _is_vec(state, 19)
                        or any(abs(v) > 1.001 for v in act[:6])
                        or act[6] not in (0.0, 1.0)):
                    raise ValueError(f"{name}:{idx} invalid request label/state")
                images = row.get("images_bgr8_npy", {})
                for role in ("base", "wrist"):
                    rel = images.get(role)
                    if (not isinstance(rel, str) or Path(rel).is_absolute()
                            or ".." in Path(rel).parts
                            or not rel.startswith(f"frames/{role}/")):
                        raise ValueError(f"{name}:{idx} unsafe image path")
                    if not (ep / rel).is_file():
                        raise FileNotFoundError(f"{name}:{idx} image missing: {rel}")
                local_count += 1
            segments.append({
                "episode": name, "start": start, "stop": stop,
                "ends_original_episode": stop == len(rows),
            })
        if local_count > info["counts"].get("complete_7d_request", 0):
            raise ValueError(f"{name}: exporter count exceeds audited count")
        total += local_count
    expected = q[split]["complete_7d_transitions"]
    if total != expected:
        # This protects the current one-run/episode dataset from losing
        # short valid regions or inadvertently joining disjoint windows.
        raise ValueError(
            f"Split has {expected} valid actions but only {total} in >= "
            f"{min_horizon}-length spans; revise quality contract before export"
        )
    if not isinstance(task, str) or not task.strip() or fps != 10:
        raise ValueError("Expected one named 10 Hz capture task")
    for role in ("base", "wrist"):
        shape = profiles.get(role, {}).get("resolution")
        if not (isinstance(shape, list) and len(shape) == 2
                and all(type(v) is int and v > 0 for v in shape)):
            raise ValueError("Invalid camera dimensions")
    return {
        "candidate_root": str(candidate),
        "quality_sha256": hashlib.sha256(quality.read_bytes()).hexdigest(),
        "quality_manifest": str(quality),
        "source_root": str(raw),
        "split": split,
        "action_labels": total,
        "source_episodes": len(selected),
        "segments": segments,
        "task": task,
        "camera_profiles": profiles,
        "fps": fps,
        "horizon": min_horizon,
        "sft_ready": False, "robot_execution_approved": False,
    }


def export(pre: dict, output: Path, *, repo_id: str) -> None:
    output = output.expanduser().resolve()
    raw = Path(pre["source_root"])
    candidate = Path(pre["candidate_root"])
    if (output.exists() or output == raw or output.is_relative_to(raw)
            or output == candidate or output.is_relative_to(candidate)):
        raise ValueError("Output must be NEW and outside both source datasets")
    if "/" not in repo_id or not repo_id.strip():
        raise ValueError("Explicit local/owner LeRobot repo id required")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from rlinf.data.storage.lerobot.compat import add_frame_to_dataset

    camera = pre["camera_profiles"]
    features = {
        "image": {
            "dtype": "image",
            "shape": (camera["base"]["resolution"][1],
                      camera["base"]["resolution"][0], 3),
            "names": ["height", "width", "channel"],
        },
        "extra_view_image": {
            "dtype": "image",
            "shape": (camera["wrist"]["resolution"][1],
                      camera["wrist"]["resolution"][0], 3),
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (19,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
        "is_success": {"dtype": "bool", "shape": (1,), "names": ["is_success"]},
    }
    kwargs = {
        "repo_id": repo_id, "root": output,
        "fps": pre["fps"], "robot_type": "fr3_robotiq",
        "features": features,
        "image_writer_threads": 1, "image_writer_processes": 0,
    }
    if "use_videos" in inspect.signature(LeRobotDataset.create).parameters:
        kwargs["use_videos"] = False
    dataset = LeRobotDataset.create(**kwargs)
    cached_name = None
    cached = None
    try:
        for segment in pre["segments"]:
            name = segment["episode"]
            if name != cached_name:
                cached = _jsonl(candidate / "episodes" / (name + ".jsonl"))
                cached_name = name
            for i in range(segment["start"], segment["stop"]):
                row = cached[i]
                images = {}
                for role, key in (("base", "image"), ("wrist", "extra_view_image")):
                    bgr = np.load(
                        raw / name / row["images_bgr8_npy"][role],
                        allow_pickle=False
                    )
                    expected = features[key]["shape"]
                    if bgr.shape != expected or bgr.dtype != np.uint8:
                        raise ValueError(f"{name}:{i}: unexpected {role} image")
                    images[key] = np.ascontiguousarray(bgr[:, :, ::-1])
                terminal = i == segment["stop"] - 1
                add_frame_to_dataset(dataset, {
                    **images,
                    "state": np.asarray(row["state19_candidate"], dtype=np.float32),
                    "actions": np.asarray(row["candidate_request_action_7d"], dtype=np.float32),
                    "done": np.asarray([terminal], dtype=np.bool_),
                    "is_success": np.asarray([
                        terminal and segment["ends_original_episode"]
                    ], dtype=np.bool_),
                    "task": pre["task"],
                })
            dataset.save_episode()
            print(f"Exported {name} [{segment['start']}:{segment['stop']}]", flush=True)
        if getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.stop()
            dataset.image_writer = None
        del dataset

    (output / "fr3_command_request_provenance.json").write_text(
        json.dumps({
            "schema": EXPORT_SCHEMA, "repo_id": repo_id,
            "source_split": pre["split"],
            "source_candidate_root": pre["candidate_root"],
            "quality_sha256": pre["quality_sha256"],
            "source_episodes": pre["source_episodes"],
            "frames_exported": pre["action_labels"],
            "segments": pre["segments"],
            "action_fields": list(FIELDS),
            "label_source": "time-averaged normalized SpaceMouse requests, NOT accepted targets",
            "gripper_source": "last button request, not measured knuckle angle",
            "source_action_horizon": pre["horizon"],
            "gripper_unknown_imputed": False,
            "physical_command_mapping_validated": False,
            "camera_exposure_latency_validated": False,
            "sft_ready": False, "recap_ready": False,
            "robot_execution_approved": False,
            "note": "Offline research candidate. Incompatible with measured-motion proxy model and normalization.",
        }, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-root", required=True, type=Path)
    p.add_argument("--quality", required=True, type=Path)
    p.add_argument("--split", required=True, choices=("train", "validation"))
    p.add_argument("--output", type=Path)
    p.add_argument("--repo-id", default="local/fr3_peg_spacemouse_request_candidate_v1")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    if not args.validate_only and args.output is None:
        p.error("--output required unless --validate-only")
    pre = preflight(args.candidate_root, args.quality, args.split)
    print(f"Validated {pre['split']}: {pre['source_episodes']} episodes, "
          f"{pre['action_labels']} candidate request labels")
    if args.validate_only:
        print("Read-only preflight complete; no LeRobot export or training performed.")
        return 0
    export(pre, args.output, repo_id=args.repo_id)
    print(f"Experimental LeRobot candidate written: {args.output}")
    print("Request labels NOT verified controller actions. No SFT/RECAP or robot approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
