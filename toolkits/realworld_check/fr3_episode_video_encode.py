# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Generate browser-ready, timestamp-aware MP4 previews from RAW FR3 episodes.

Original frames/ states.jsonl and operator_commands.jsonl are NEVER modified.
Two 10-ish Hz video streams retain the capture's *actual* recorded frame
times using ffmpeg's variable-frame-rate concat demuxer. The two streams
therefore have the same timeline as the dashboard's telemetry plots.

Example (one episode first):
    python -m toolkits.realworld_check.fr3_episode_video_encode \
      --root /data1/Muttaqien/fr3_peg_100 --episode 1
Batch after verifying video quality:
    python -m toolkits.realworld_check.fr3_episode_video_encode \
      --root /data1/Muttaqien/fr3_peg_100 --all

Requires ffmpeg with libx264, numpy, Pillow. Output defaults to a sibling:
    /data1/Muttaqien/fr3_peg_100_web_videos/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

EPISODE_RE = re.compile(r"episode_[0-9]{4,}")
VIDEO_SCHEMA = "fr3_browser_video_cache_v1"


def default_video_root(raw_root: Path) -> Path:
    return raw_root.parent / f"{raw_root.name}_web_videos"


def source_signature(episode: Path) -> dict:
    state = (episode / "states.jsonl").stat()
    manifest = (episode / "manifest.json").stat()
    return {
        "state_size": state.st_size,
        "state_mtime_ns": state.st_mtime_ns,
        "manifest_size": manifest.st_size,
        "manifest_mtime_ns": manifest.st_mtime_ns,
    }


def video_ready(raw_episode: Path, video_episode: Path) -> bool:
    if video_episode.is_symlink() or not video_episode.is_dir():
        return False
    for role in ("base", "wrist"):
        f = video_episode / f"{role}.mp4"
        if f.is_symlink() or not f.is_file() or not f.stat().st_size:
            return False
    try:
        meta = json.loads((video_episode / "video_manifest.json").read_text(encoding="utf-8"))
        return (meta.get("schema") == VIDEO_SCHEMA
                and meta.get("source_signature") == source_signature(raw_episode)
                and meta.get("frames", 0) > 0)
    except (OSError, ValueError, TypeError):
        return False


def episode_paths(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if EPISODE_RE.fullmatch(p.name)
        and p.is_dir() and not p.is_symlink()
        and (p / "states.jsonl").is_file()
        and (p / "manifest.json").is_file()
    )


def _rows(episode: Path) -> tuple[list[dict], list[int]]:
    with (episode / "states.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(s) for s in f if s.strip()]
    if not rows:
        raise ValueError(f"{episode.name}: no states")
    stamps = [int(row["recorded_monotonic_ns"]) for row in rows]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError(f"{episode.name}: timestamps must increase strictly")
    return rows, stamps


def _encode_role(episode: Path, rows: list[dict], stamps: list[int],
                 role: str, temp_dir: Path, *, quality: int, crf: int,
                 ffmpeg: str) -> None:
    frame_dir = temp_dir / f"{role}_jpg"
    frame_dir.mkdir()
    concat_file = frame_dir / "frames.ffconcat"
    durations = [
        (b - a) * 1e-9 for a, b in zip(stamps, stamps[1:])
    ]
    # Give the final image a short display hold so the browser can seek to
    # the last original frame. This does not change its original timestamp.
    last_hold = float(np.median(durations)) if durations else 0.1

    with concat_file.open("w", encoding="utf-8") as timeline:
        timeline.write("ffconcat version 1.0\n")
        for index, row in enumerate(rows):
            relative = row.get("image_files_bgr8", {}).get(role)
            if (not isinstance(relative, str)
                or not re.fullmatch(r"frames/" + role + r"/[0-9]{6}\.npy", relative)):
                raise ValueError(f"{episode.name}: unsafe/missing {role} frame reference")
            npy = (episode / relative).resolve(strict=True)
            if not npy.is_relative_to(episode.resolve()):
                raise ValueError("Frame escapes original episode directory")
            array = np.load(npy, allow_pickle=False)
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise ValueError(f"Invalid camera frame {npy}: {array.shape}/{array.dtype}")
            jpg = frame_dir / f"{index:06d}.jpg"
            Image.fromarray(np.ascontiguousarray(array[..., ::-1])).save(
                jpg, format="JPEG", quality=quality, subsampling=0,
            )
            timeline.write(f"file '{index:06d}.jpg'\n")
            # 1000 Hz demuxer timebase avoids default 25 Hz (=40 ms) rounding.
            timeline.write("option framerate 1000\n")
            timeline.write(f"duration {durations[index] if index < len(durations) else last_hold:.9f}\n")
        # Repeat the final file entry to preserve its hold duration in MP4.
        timeline.write(f"file '{len(rows)-1:06d}.jpg'\noption framerate 1000\n")

    output = temp_dir / f"{role}.mp4"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-fps_mode", "vfr", "-c:v", "libx264", "-preset", "veryfast",
        "-threads", "2", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-video_track_timescale", "1000000", "-movflags", "+faststart",
        str(output),
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {episode.name}/{role}: {run.stderr[-2000:]}"
        )
    if not output.is_file() or output.stat().st_size < 1000:
        raise RuntimeError(f"No usable video produced for {episode.name}/{role}")

    # Remove temporary images as soon as this stream is encoded.
    shutil.rmtree(frame_dir)


def encode(episode: Path, dest_root: Path, *, quality: int = 86,
           crf: int = 25, force: bool = False, ffmpeg: str = "ffmpeg") -> str:
    if not EPISODE_RE.fullmatch(episode.name) or not episode.is_dir():
        raise ValueError("Expected a valid episode directory")
    episode = episode.resolve(strict=True)
    dest_root = dest_root.resolve()
    if dest_root.is_relative_to(episode) or episode.is_relative_to(dest_root):
        raise ValueError("Video cache must be outside raw episode directory")
    dest_root.mkdir(parents=True, exist_ok=True)
    target = dest_root / episode.name
    if video_ready(episode, target) and not force:
        return "cached"
    if target.exists() and not force:
        raise FileExistsError(f"Cache exists but is incomplete/stale: {target}. Use --force.")

    rows, stamps = _rows(episode)
    signature = source_signature(episode)
    with tempfile.TemporaryDirectory(prefix=f".{episode.name}_", dir=dest_root) as tmp:
        tmp_dir = Path(tmp)
        for role in ("base", "wrist"):
            print(f"{episode.name}: encoding {role} ({len(rows)} frames)", flush=True)
            _encode_role(episode, rows, stamps, role, tmp_dir,
                         quality=quality, crf=crf, ffmpeg=ffmpeg)
        durations = [(b - a) * 1e-9 for a, b in zip(stamps, stamps[1:])]
        metadata = {
            "schema": VIDEO_SCHEMA,
            "episode": episode.name,
            "frames": len(rows),
            "duration_s": (stamps[-1] - stamps[0]) * 1e-9,
            "video_end_hold_s": float(np.median(durations)) if durations else 0.1,
            "source_signature": signature,
            "capture_timestamps": "recorded_monotonic_ns",
            "encoded_timing": "VFR, millisecond-level timebase",
            "original_raw_episodes_modified": False,
        }
        (tmp_dir / "video_manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if target.exists():
            shutil.rmtree(target)
        os.rename(tmp_dir, target)
    return "encoded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data1/Muttaqien/fr3_peg_100"))
    parser.add_argument("--video-root", type=Path, default=None)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode", type=str, help="Number or episode_0001")
    selection.add_argument("--all", action="store_true", help="Encode all raw episodes")
    parser.add_argument("--force", action="store_true", help="Replace old video cache")
    parser.add_argument("--crf", type=int, default=25)
    parser.add_argument("--jpeg-quality", type=int, default=86)
    args = parser.parse_args()

    if not 0 <= args.crf <= 51 or not 1 <= args.jpeg_quality <= 100:
        parser.error("CRF must be 0-51 and jpeg quality 1-100")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg executable is required; install ffmpeg on the lab machine")
    root = args.root.expanduser().resolve(strict=True)
    video_root = (args.video_root.expanduser().resolve()
                  if args.video_root else default_video_root(root))
    if video_root == root or video_root.is_relative_to(root):
        parser.error("Video root must not be inside the raw dataset root")
    paths = episode_paths(root)
    if not args.all:
        value = args.episode
        name = value if value.startswith("episode_") else f"episode_{int(value):04d}"
        paths = [p for p in paths if p.name == name]
        if not paths:
            parser.error(f"Episode not found: {name}")
    print(f"Generating MP4 cache in {video_root}, {len(paths)} episode(s)", flush=True)
    for episode in paths:
        result = encode(episode, video_root, quality=args.jpeg_quality,
                        crf=args.crf, force=args.force)
        print(f"{episode.name}: {result}", flush=True)


if __name__ == "__main__":
    main()
