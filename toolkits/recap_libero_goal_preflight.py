"""Read-only preflight for task-specific LIBERO-Goal Task 0 RECAP datasets.

Run before launching GPU training:
    python -m toolkits.recap_libero_goal_preflight
    python -m toolkits.recap_libero_goal_preflight --stage returns
    python -m toolkits.recap_libero_goal_preflight --stage advantages

This checks real LeRobot Parquet action, camera, success, episode and task metadata.
A 20-episode visualization gallery does not satisfy this preflight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from toolkits.recap_libero_goal_collect import DEFAULT_DATA_ROOT, SPLITS
from examples.offline_rl.advantage_labeling.recap.process.compute_returns import parse_episode_success

TAG = "libero_goal_task0_recap"
REQUIRED = ("image", "wrist_image", "state", "actions", "is_success", "episode_index", "frame_index")
CHECKED_SUITES = ("libero_goal",)


def _dataset_episodes(dataset: Path) -> tuple[int, int, int, str]:
    """Validate all frame metadata from Parquet without loading video pixels."""
    import pyarrow.parquet as pq

    parquet_files = sorted((dataset / "data").rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"{dataset}: no LeRobot data parquet files")
    episodes: dict[int, dict] = {}
    prompt = ""
    columns_to_check = ("episode_index", "frame_index", "is_success", "task_index")
    for file in parquet_files:
        schema = pq.ParquetFile(file).schema_arrow
        if not set(columns_to_check[:-1]).issubset(set(schema.names)):
            raise ValueError(f"{file}: missing episode/frame/success columns")
        cols = [x for x in columns_to_check if x in schema.names]
        table = pq.read_table(file, columns=cols)
        data = table.to_pydict()
        for index, frame_index, flag in zip(data["episode_index"], data["frame_index"], data["is_success"]):
            ep = int(index)
            frame = int(frame_index)
            success = parse_episode_success(flag)
            meta = episodes.setdefault(ep, {"frames": set(), "success": success})
            if frame in meta["frames"] or meta["success"] != success:
                raise ValueError(f"{file}: inconsistent frame index or success flag in episode {ep}")
            meta["frames"].add(frame)
        if "task_index" in data and any(int(i) != 0 for i in data["task_index"]):
            raise ValueError(f"{file}: this dataset contains multiple task indices")
    if not episodes or sorted(episodes) != list(range(len(episodes))):
        raise ValueError(f"{dataset}: episode indices must be contiguous from zero")
    for ep, meta in episodes.items():
        if meta["frames"] != set(range(len(meta["frames"]))):
            raise ValueError(f"{dataset}: episode {ep} has missing frame indices")

    task_file = dataset / "meta" / "tasks.jsonl"
    if task_file.is_file():
        tasks = [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(tasks) != 1:
            raise ValueError(f"{task_file}: expected exactly one task instruction")
        prompt = str(tasks[0].get("task", ""))
    successes = sum(int(meta["success"]) for meta in episodes.values())
    frames = sum(len(meta["frames"]) for meta in episodes.values())
    return len(episodes), successes, frames, prompt


def check_datasets(root: Path, stage: str = "data") -> list[str]:
    """Return actionable errors. Does not create or change any files."""
    import pyarrow.parquet as pq

    problems = []
    prompts = set()
    seeds = []
    for split, (name, seed_start) in SPLITS.items():
        dataset = root / name
        info_file = dataset / "meta/info.json"
        manifest_file = dataset / "collection_manifest.json"
        if not info_file.is_file() or not manifest_file.is_file():
            problems.append(f"{split}: missing complete Goal Task 0 dataset at {dataset}")
            continue
        try:
            info = json.loads(info_file.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("suite") != "libero_goal" or manifest.get("task_index") != 0 or manifest.get("split") != split:
                raise ValueError("split/task identity mismatch")
            if manifest.get("seed_start") != seed_start:
                raise ValueError("unexpected split seed; data may overlap")
            seeds.append(seed_start)
            features = info.get("features", {})
            missing = [name for name in REQUIRED[:5] if name not in features]
            if missing:
                raise ValueError(f"missing LeRobot features: {missing}")
            if tuple(features["state"].get("shape", ())) != (8,) or tuple(features["actions"].get("shape", ())) != (7,):
                raise ValueError("expected LIBERO state (8,) and actions (7,)")
            expected = int(manifest["episodes"])
            total, successes, frames, prompt = _dataset_episodes(dataset)
            if total != expected or total != int(info.get("total_episodes", -1)):
                raise ValueError(f"episode-count mismatch: Parquet={total}, collection={expected}, LeRobot={info.get('total_episodes')}")
            if split == "sft" and successes != total:
                raise ValueError("SFT dataset includes failed episodes")
            if split in ("train", "eval") and not (0 < successes < total):
                raise ValueError("rollout dataset requires both successful and failed episodes")
            if successes != int(manifest.get("successes", -1)):
                raise ValueError("success counts differ from collector manifest")
            if not prompt or prompt != manifest.get("task_description"):
                raise ValueError("task instruction differs from collection manifest")
            prompts.add(prompt)
            if stage in ("returns", "advantages"):
                returns_path = dataset / "meta" / f"returns_{TAG}.parquet"
                if not returns_path.is_file():
                    raise ValueError(f"return sidecar missing: {returns_path}")
                returned = pq.ParquetFile(returns_path).metadata.num_rows
                if returned != frames:
                    raise ValueError(f"return sidecar has {returned} rows, expected {frames}")
            if split == "train" and stage == "advantages":
                adv_path = dataset / "meta" / f"advantages_{TAG}.parquet"
                if not adv_path.is_file():
                    raise ValueError(f"advantage sidecar missing: {adv_path}")
                adv = pq.ParquetFile(adv_path)
                columns = set(adv.schema_arrow.names)
                if not {"advantage", "episode_index", "frame_index"}.issubset(columns):
                    raise ValueError("advantage sidecar missing required columns")
                if adv.metadata.num_rows != frames:
                    raise ValueError("advantage sidecar row count differs from rollout frames")
            print(f"PASS {split}: {total} episodes, {successes} successes, {frames} frames", flush=True)
        except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as error:
            problems.append(f"{split}: {error}")
    if len(seeds) == len(SPLITS) and len(set(seeds)) != len(SPLITS):
        problems.append("Split seed ranges overlap")
    if len(prompts) > 1:
        problems.append("Datasets contain different language task instructions")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stage", choices=("data", "returns", "advantages"), default="data")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        print(f"FAIL: Goal Task 0 data directory missing: {root}", flush=True)
        return 1
    try:
        problems = check_datasets(root, args.stage)
    except ImportError as error:
        print(f"FAIL: missing dependency: {error}", flush=True)
        return 1
    for problem in problems:
        print(f"FAIL: {problem}", flush=True)
    if problems:
        return 1
    print(f"PASS: LIBERO-Goal Task 0 datasets ready for {args.stage}; no training was performed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
