# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Record 20 REAL, supervised pi0.5 FR3 rollout outcomes for RECAP.

This is an outcome registrar, not a motion controller. Each episode must have
physical execution evidence (measured states and issued policy commands) before
it can be counted. No ROS publisher is opened; no false success is inferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET = 20
MAX_ATTEMPTS = 9999
OUTCOMES = {"success", "failure", "aborted"}
REQUIRED = {"episode_id", "policy_checkpoint", "policy_command_log",
            "measured_state_log", "outcome", "operator_reviewed"}


def validate_episode(entry: dict, *, root: Path) -> dict:
    if not isinstance(entry, dict) or not REQUIRED.issubset(entry):
        raise ValueError(f"Missing episode fields: {sorted(REQUIRED - set(entry))}")
    if not isinstance(entry["episode_id"], int) or not 1 <= entry["episode_id"] <= MAX_ATTEMPTS:
        raise ValueError("episode_id must be 1..9999")
    if entry["outcome"] not in OUTCOMES:
        raise ValueError("outcome must be success, failure, or aborted")
    if entry["operator_reviewed"] is not True:
        raise ValueError("Outcome must be operator-reviewed")
    if "global_step_3000" not in entry["policy_checkpoint"]:
        raise ValueError("Not the trained pi0.5 SFT checkpoint")
    for key in ("policy_command_log", "measured_state_log"):
        name = entry[key]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Missing {key}")
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{key} missing or empty: {path}")
    if entry.get("shadow_only") or entry.get("robot_commands_sent", 0) <= 0:
        raise ValueError("Shadow-only/no-command observations are not real rollouts")
    return entry


def progress(root: Path) -> dict:
    root = root.resolve()
    found = sorted(root.glob("episode_*/outcome.json"))
    episodes = []
    for file in found:
        episode = validate_episode(json.loads(file.read_text()), root=file.parent)
        if file.parent.name != f"episode_{episode['episode_id']:02d}":
            raise ValueError("Episode directory and ID do not agree")
        episodes.append(episode)
    ids = [item["episode_id"] for item in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode IDs")
    # Never overwrite a partially recorded physical episode after a fault.
    occupied_ids = {
        int(folder.name[8:]) for folder in root.glob("episode_*")
        if folder.is_dir() and folder.name[8:].isdigit()
        and 1 <= int(folder.name[8:]) <= MAX_ATTEMPTS
        and any(folder.iterdir())
    }
    return {
        "real_rollouts": sum(x["outcome"] in ("success", "failure") for x in episodes),
        "recorded_attempts": len(episodes),
        "incomplete_attempts": len(occupied_ids - set(ids)),
        "target": TARGET,
        "successes": sum(x["outcome"] == "success" for x in episodes),
        "failures": sum(x["outcome"] == "failure" for x in episodes),
        "aborted": sum(x["outcome"] == "aborted" for x in episodes),
        "next_episode": (None if sum(x["outcome"] in ("success", "failure") for x in episodes) >= TARGET
                         else next((i for i in range(1, MAX_ATTEMPTS + 1)
                                    if i not in occupied_ids), None)),
        "outcomes_complete": sum(x["outcome"] in ("success", "failure") for x in episodes) == TARGET,
        "recap_dataset_ready": False,  # Export and real transition validation still required.
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("/data1/Muttaqien/fr3_pi05_real_rollouts_20"))
    p.add_argument("--record", type=Path, help="Existing operator-labeled outcome.json")
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.record:
        item = json.loads(args.record.read_text(encoding="utf-8"))
        validate_episode(item, root=args.record.parent)
        folder = args.root / f"episode_{item['episode_id']:02d}"
        folder.mkdir(exist_ok=True)
        destination = folder / "outcome.json"
        if destination.exists():
            raise FileExistsError(f"Episode already registered: {destination}")
        destination.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(progress(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
