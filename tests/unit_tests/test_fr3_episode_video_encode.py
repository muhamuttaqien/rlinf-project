"""Offline video cache tests; never connect to real robot hardware."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from toolkits.realworld_check.fr3_episode_video_encode import (
    encode, source_signature, video_ready, VIDEO_SCHEMA,
)
from toolkits.realworld_check.fr3_episode_web_dashboard import EpisodeStore


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="ffmpeg/ffprobe required for browser MP4 encoding")
def test_vfr_two_camera_cache_keeps_original_frames(tmp_path):
    episode = tmp_path / "raw" / "episode_0001"
    (episode / "frames" / "base").mkdir(parents=True)
    (episode / "frames" / "wrist").mkdir(parents=True)
    rows = []
    timestamps = [1_000_000_000, 1_100_000_000, 1_370_000_000]
    for i, timestamp in enumerate(timestamps):
        paths = {}
        for role in ("base", "wrist"):
            rel = f"frames/{role}/{i:06d}.npy"
            array = np.full((24, 32, 3), i * 60 + 20, dtype=np.uint8)
            np.save(episode / rel, array)
            paths[role] = rel
        rows.append({
            "frame_index": i,
            "recorded_monotonic_ns": timestamp,
            "measured_state": {},
            "image_files_bgr8": paths,
        })
    (episode / "states.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (episode / "manifest.json").write_text(json.dumps({
        "status": "raw_recorded", "frames_saved": 3,
        "sample_rate_hz_requested": 10, "task": "synthetic"
    }), encoding="utf-8")
    (episode / "operator_commands.jsonl").write_text("", encoding="utf-8")
    signature = source_signature(episode)
    video_root = tmp_path / "video_cache"
    assert encode(episode, video_root) == "encoded"
    target = video_root / "episode_0001"
    assert video_ready(episode, target)
    assert encode(episode, video_root) == "cached"
    metadata = json.loads((target / "video_manifest.json").read_text())
    assert metadata["schema"] == VIDEO_SCHEMA
    assert metadata["frames"] == 3
    assert metadata["duration_s"] == pytest.approx(0.37)
    for role in ("base", "wrist"):
        out = target / f"{role}.mp4"
        assert out.is_file() and out.stat().st_size > 1000
        frames = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time",
            "-of", "csv=p=0", str(out),
        ], check=True, capture_output=True, text=True)
        times = []
        for line in frames.stdout.splitlines():
            stripped = line.strip().strip(",")
            if stripped:
                times.append(float(stripped))
        # ffmpeg's JPEG concat demuxer rounds to nearest millisecond.
        assert times[:3] == pytest.approx([0, 0.1, 0.37], abs=0.003)

    store = EpisodeStore(tmp_path / "raw", video_root=video_root)
    assert store.video_path("episode_0001", "base") == target / "base.mp4"
    assert store.video_path("episode_0001", "wrist") == target / "wrist.mp4"
    assert source_signature(episode) == signature
    assert len(list(episode.glob("frames/base/*.npy"))) == 3


def test_video_ready_rejects_stale_and_incomplete_cache(tmp_path):
    raw = tmp_path / "episode_0001"
    raw.mkdir()
    (raw / "states.jsonl").write_text("{}\n")
    (raw / "manifest.json").write_text("{}")
    cache = tmp_path / "cache" / "episode_0001"
    cache.mkdir(parents=True)
    assert not video_ready(raw, cache)
    for role in ("base", "wrist"):
        (cache / f"{role}.mp4").write_bytes(b"video")
    (cache / "video_manifest.json").write_text(json.dumps({
        "schema": VIDEO_SCHEMA, "frames": 1,
        "source_signature": source_signature(raw),
    }))
    assert video_ready(raw, cache)
    (raw / "states.jsonl").write_text('{"modified": true}\n')
    assert not video_ready(raw, cache)
