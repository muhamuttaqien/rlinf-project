# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Validate pilot StackCube LeRobot transitions against their raw traces."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Collector output root"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    dataset_root = root / "dataset"
    info_path = dataset_root / "meta" / "info.json"
    summaries_path = root / "episode_summaries.json"
    if not info_path.is_file() or not summaries_path.is_file():
        raise FileNotFoundError(
            "LeRobot metadata or collector episode summaries are missing"
        )
    info = json.loads(info_path.read_text())
    summaries = json.loads(summaries_path.read_text())
    if info["total_episodes"] != len(summaries):
        raise ValueError("LeRobot episode count does not match collector summaries")
    if info["robot_type"] != "panda_wristcam" or info["fps"] != 20:
        raise ValueError("Unexpected LeRobot robot type or frame rate")

    features = info["features"]
    total = 0
    trace_verified = 0
    decode_verified = 0
    expected_keys = (
        "observation.state",
        "action",
        "policy_action",
        "reward",
        "is_success",
        "terminated",
        "truncated",
        "grasped",
        "lifted_by_2_5cm",
        "transported_by_3cm",
        "simulator_step",
        "inference_call",
        "action_index_in_chunk",
        "observation.images.base_camera",
        "observation.images.hand_camera",
    )
    missing = set(expected_keys) - set(features)
    if missing:
        raise ValueError(f"Missing dataset features: {sorted(missing)}")

    offset = 0
    for episode_index, summary in enumerate(summaries):
        parquet = (
            dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(parquet)
        trace_path = root / "traces" / f"seed_{summary['seed']}.jsonl"
        trace_rows = [
            json.loads(line) for line in trace_path.read_text().splitlines() if line
        ]
        length = int(summary["steps"])
        if table.num_rows != length or len(trace_rows) != length:
            raise ValueError(
                f"Episode {episode_index}: parquet/trace/summary lengths disagree"
            )
        if length < 1 or length > 150:
            raise ValueError(f"Episode {episode_index}: invalid length {length}")
        for column in ("observation.state", "action", "policy_action", "reward"):
            values = np.stack(table.column(column).to_pylist())
            if not np.isfinite(values).all():
                raise ValueError(f"Episode {episode_index}: non-finite {column}")
        action = np.stack(table.column("action").to_pylist())
        policy_action = np.stack(table.column("policy_action").to_pylist())
        if action.shape != (length, 7) or not np.array_equal(action, policy_action):
            raise ValueError(
                f"Episode {episode_index}: action shape or policy/executed action mismatch"
            )
        states = np.stack(table.column("observation.state").to_pylist())
        rewards = np.asarray(
            table.column("reward").to_pylist(), dtype=np.float32
        ).reshape(-1)
        success = np.asarray(
            table.column("is_success").to_pylist(), dtype=bool
        ).reshape(-1)
        terminated = np.asarray(
            table.column("terminated").to_pylist(), dtype=bool
        ).reshape(-1)
        truncated = np.asarray(
            table.column("truncated").to_pylist(), dtype=bool
        ).reshape(-1)
        steps = np.asarray(
            table.column("simulator_step").to_pylist(), dtype=np.int64
        ).reshape(-1)
        if states.shape != (length, 18) or not np.array_equal(
            steps, np.arange(1, length + 1)
        ):
            raise ValueError(
                f"Episode {episode_index}: state shape or simulator steps invalid"
            )
        if bool(success[-1]) != bool(summary["success"]):
            raise ValueError(f"Episode {episode_index}: final native success disagrees")
        if bool(terminated[-1]) != bool(summary["terminated"]) or bool(
            truncated[-1]
        ) != bool(summary["truncated"]):
            raise ValueError(f"Episode {episode_index}: terminal flags disagree")
        if np.any(terminated[:-1]) or np.any(truncated[:-1]) or np.any(success[:-1]):
            raise ValueError(
                f"Episode {episode_index}: data continued after a terminal transition"
            )
        trace_actions = np.asarray(
            [row["executed_action"] for row in trace_rows], dtype=np.float32
        )
        trace_states = np.asarray(
            [row["state_pre_action"] for row in trace_rows], dtype=np.float32
        )
        trace_rewards = np.asarray(
            [row["reward"] for row in trace_rows], dtype=np.float32
        )
        if not np.array_equal(action, trace_actions) or not np.allclose(
            states, trace_states, atol=1e-6
        ):
            raise ValueError(
                f"Episode {episode_index}: stored action/state differs from raw trace"
            )
        if not np.allclose(rewards, trace_rewards, atol=1e-6):
            raise ValueError(
                f"Episode {episode_index}: stored rewards differ from raw trace"
            )
        trace_verified += length
        for camera in (
            "observation.images.base_camera",
            "observation.images.hand_camera",
        ):
            video_path = (
                dataset_root
                / "videos"
                / "chunk-000"
                / camera
                / f"episode_{episode_index:06d}.mp4"
            )
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            if (stream.width, stream.height, stream.frames) != (128, 128, length):
                raise ValueError(
                    f"Episode {episode_index}: video metadata does not match transitions"
                )
            selected = {0, length // 2, length - 1}
            seen = set()
            for frame_index, video_frame in enumerate(container.decode(video=0)):
                if frame_index in selected:
                    image = video_frame.to_ndarray(format="rgb24")
                    if image.shape != (128, 128, 3) or image.max() == image.min():
                        raise ValueError(
                            f"Episode {episode_index}: invalid decoded image {camera}"
                        )
                    seen.add(frame_index)
            container.close()
            if seen != selected:
                raise ValueError(
                    f"Episode {episode_index}: missing sampled video frames {camera}"
                )
            decode_verified += len(selected)
        total += length
        offset += length

    if total != info["total_frames"]:
        raise ValueError("Total transition count disagrees with dataset metadata")
    report = {
        "status": "PASSED",
        "episodes": len(summaries),
        "transitions": total,
        "trace_transition_rows_verified": trace_verified,
        "decoded_video_samples_verified": decode_verified,
        "successes": sum(bool(row["success"]) for row in summaries),
        "terminal_success_flags": sum(bool(row["success"]) for row in summaries),
        "episode_lengths": [int(row["steps"]) for row in summaries],
        "dataset_root": str(dataset_root),
    }
    (root / "integrity_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
