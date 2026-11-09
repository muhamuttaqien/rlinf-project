# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Read-only FR3 recorded-episode browser dashboard.

Serves saved raw .npy BGR frames, measured FR3 state, and timestamped ROS
operator events. Does NOT import robotics drivers, connect to ROS, or issue
robot commands. Binds to loopback ONLY: use SSH port forwarding for home use.

Run:
    python -m toolkits.realworld_check.fr3_episode_web_dashboard \
      --root /data1/Muttaqien/fr3_peg_100 --port 8766
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

EPISODE_RE = re.compile(r"episode_[0-9]{4,}")
FRAME_ROUTE = re.compile(r"/api/frame/(episode_[0-9]{4,})/(base|wrist)/([0-9]+)")
EPISODE_ROUTE = re.compile(r"/api/episode/(episode_[0-9]{4,})")

STATE_FIELDS = {
    "tcp_pose": ("tcp_x", "tcp_y", "tcp_z", "quat_x", "quat_y", "quat_z", "quat_w"),
    "arm_joint_position": tuple(f"q{i}" for i in range(1, 8)),
    "arm_joint_velocity": tuple(f"dq{i}" for i in range(1, 8)),
    "arm_joint_effort": tuple(f"tau{i}" for i in range(1, 8)),
    "tcp_force": ("fx", "fy", "fz"),
    "tcp_torque": ("tx", "ty", "tz"),
    "robotiq_knuckle_position": ("gripper_angle",),
}
OPERATOR_FIELDS = (
    "op_lin_x", "op_lin_y", "op_lin_z", "op_ang_x", "op_ang_y", "op_ang_z",
    "op_grip_request",
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        obj = json.load(stream)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


class EpisodeStore:
    """Reads only files within the explicitly configured raw dataset root."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    def episode_dir(self, name: str) -> Path:
        if not EPISODE_RE.fullmatch(name):
            raise ValueError("Invalid episode identifier")
        path = self.root / name
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != self.root:
            raise FileNotFoundError(name)
        return path

    def episodes(self) -> dict:
        result = []
        for path in sorted(self.root.iterdir()):
            if not EPISODE_RE.fullmatch(path.name) or not path.is_dir() or path.is_symlink():
                continue
            try:
                manifest = _read_json(path / "manifest.json")
                review_path = path / "episode_review.json"
                review = _read_json(review_path) if review_path.is_file() else {}
                result.append({
                    "name": path.name,
                    "frames": int(manifest.get("frames_saved", 0)),
                    "fps": int(manifest.get("sample_rate_hz_requested", 10)),
                    "task": str(manifest.get("task", "")),
                    "capture_status": str(manifest.get("status", "unknown")),
                    "outcome": str(review.get("operator_outcome", "unreviewed")),
                })
            except (ValueError, OSError, KeyError, TypeError):
                continue
        return {
            "episodes": result,
            "episode_count": len(result),
            "total_frames": sum(row["frames"] for row in result),
            "operator_successes": sum(row["outcome"] == "success" for row in result),
        }

    @lru_cache(maxsize=4)
    def load(self, name: str) -> tuple[dict, list[dict]]:
        directory = self.episode_dir(name)
        manifest = _read_json(directory / "manifest.json")
        review_path = directory / "episode_review.json"
        review = _read_json(review_path) if review_path.is_file() else {}
        rows = _read_jsonl(directory / "states.jsonl")
        if not rows:
            raise ValueError("Episode has no recorded frames")

        times_ns = [int(row["recorded_monotonic_ns"]) for row in rows]
        if any(b < a for a, b in zip(times_ns, times_ns[1:])):
            raise ValueError("Frame timestamps must be nondecreasing")
        times = [round((stamp - times_ns[0]) * 1e-9, 5) for stamp in times_ns]

        series = {
            key: [] for fields in STATE_FIELDS.values() for key in fields
        }
        series.update({key: [] for key in OPERATOR_FIELDS})
        for row in rows:
            measured = row.get("measured_state", {})
            for raw_key, names in STATE_FIELDS.items():
                values = measured.get(raw_key, [])
                if not isinstance(values, list):
                    values = []
                for index, key in enumerate(names):
                    series[key].append(_finite(values[index]) if index < len(values) else None)

        commands_file = directory / "operator_commands.jsonl"
        commands = _read_jsonl(commands_file) if commands_file.is_file() else []
        commands = sorted(
            [cmd for cmd in commands if isinstance(cmd, dict)
             and isinstance(cmd.get("received_monotonic_ns"), int)],
            key=lambda row: row["received_monotonic_ns"],
        )
        # Mirror the original fr3_visualize.py operator plot: use discrete
        # ROS event times, twist vector magnitudes, and requested gripper values.
        # These are NOT verified executed robot-controller commands.
        events = {
            "twist": {"times_s": [], "values": []},
            "gripper": {"times_s": [], "values": []},
        }
        for cmd in commands:
            t = (cmd["received_monotonic_ns"] - times_ns[0]) * 1e-9
            message = cmd.get("message")
            if not isinstance(message, dict):
                continue
            if cmd.get("name") == "operator_twist":
                vectors = []
                for component in ("linear", "angular"):
                    values = message.get(component)
                    if not isinstance(values, dict):
                        break
                    vectors.extend(_finite(values.get(axis)) for axis in "xyz")
                if len(vectors) == 6 and all(v is not None for v in vectors):
                    magnitude = math.sqrt(sum(v * v for v in vectors))
                    events["twist"]["times_s"].append(round(t, 5))
                    events["twist"]["values"].append(magnitude)
            elif cmd.get("name") == "operator_gripper_request":
                requested = _finite(message.get("data"))
                if requested is not None:
                    events["gripper"]["times_s"].append(round(t, 5))
                    events["gripper"]["values"].append(requested)

        held = {key: None for key in OPERATOR_FIELDS}
        pos = 0
        for stamp in times_ns:
            while pos < len(commands) and commands[pos]["received_monotonic_ns"] <= stamp:
                event = commands[pos]
                message = event.get("message", {})
                if isinstance(message, dict) and event.get("name") == "operator_twist":
                    for component, prefix in (("linear", "op_lin"), ("angular", "op_ang")):
                        vector = message.get(component, {})
                        if isinstance(vector, dict):
                            for axis in "xyz":
                                held[f"{prefix}_{axis}"] = _finite(vector.get(axis))
                elif isinstance(message, dict) and event.get("name") == "operator_gripper_request":
                    held["op_grip_request"] = _finite(message.get("data"))
                pos += 1
            for key in OPERATOR_FIELDS:
                series[key].append(held[key])

        data = {
            "name": name,
            "task": str(manifest.get("task", "")),
            "capture_status": str(manifest.get("status", "unknown")),
            "outcome": str(review.get("operator_outcome", "unreviewed")),
            "frames": len(rows),
            "fps": int(manifest.get("sample_rate_hz_requested", 10)),
            "duration_s": times[-1],
            "times_s": times,
            "operator_event_count": len(commands),
            "operator_events": events,
            "series": series,
            "notes": (
                "Read-only measured-state playback. Operator ROS twist/gripper "
                "requests are last-received event values, not verified controller "
                "actions. Camera synchronization is based on capture receipts."
            ),
        }
        return data, rows

    @lru_cache(maxsize=96)
    def frame_jpeg(self, name: str, role: str, index: int) -> bytes:
        if role not in ("base", "wrist"):
            raise ValueError("Unknown camera")
        directory = self.episode_dir(name)
        _, rows = self.load(name)
        if index < 0 or index >= len(rows):
            raise IndexError("Frame out of bounds")
        rel = rows[index].get("image_files_bgr8", {}).get(role)
        if not isinstance(rel, str) or not re.fullmatch(
            r"frames/" + role + r"/[0-9]{6}\.npy", rel
        ):
            raise ValueError("Unexpected frame filename")
        path = (directory / rel).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise FileNotFoundError("Frame file missing")
        bgr = np.load(path, allow_pickle=False)
        if bgr.ndim != 3 or bgr.shape[-1] != 3 or bgr.dtype != np.uint8:
            raise ValueError("Expected HWC uint8 BGR")
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        image = Image.fromarray(rgb, mode="RGB")
        image.thumbnail((760, 480))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=76)
        return buffer.getvalue()


class DashboardHandler(BaseHTTPRequestHandler):
    store: EpisodeStore

    def _send(self, status: int, body: bytes, mime: str, cache: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data: blob:; "
                         "style-src 'self' 'unsafe-inline'; script-src 'self' "
                         "'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: dict):
        self._send(status, json.dumps(obj, allow_nan=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):
        route = urlsplit(self.path).path
        try:
            if route == "/":
                body = Path(__file__).with_suffix(".html").read_bytes()
                return self._send(200, body, "text/html; charset=utf-8")
            if route == "/api/episodes":
                return self._json(200, self.store.episodes())
            episode = EPISODE_ROUTE.fullmatch(route)
            if episode:
                return self._json(200, self.store.load(episode.group(1))[0])
            frame = FRAME_ROUTE.fullmatch(route)
            if frame:
                body = self.store.frame_jpeg(
                    frame.group(1), frame.group(2), int(frame.group(3))
                )
                return self._send(200, body, "image/jpeg", cache=True)
            return self._json(404, {"error": "Not found"})
        except (FileNotFoundError, IndexError, ValueError, KeyError, OSError) as error:
            self.log_error("Unavailable resource: %s", error)
            return self._json(404, {"error": "Episode or frame unavailable"})
        except Exception:
            self.log_error("Unexpected dashboard error", exc_info=True)
            return self._json(500, {"error": "Dashboard error"})

    def do_POST(self):
        self._json(405, {"error": "Read-only dashboard"})

    def log_message(self, fmt, *args):
        # Avoid printing sensitive query strings; all API endpoints are fixed.
        if not self.path.startswith("/api/frame/"):
            super().log_message(fmt, *args)


def make_handler(store: EpisodeStore):
    return type("BoundDashboardHandler", (DashboardHandler,), {"store": store})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data1/Muttaqien/fr3_peg_100"))
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    store = EpisodeStore(args.root)
    summary = store.episodes()
    if summary["episode_count"] == 0:
        parser.error(f"No recorded episodes found under {store.root}")
    # SECURITY: fixed to loopback. Never expose the raw robot footage on 0.0.0.0.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"FR3 Episode Dashboard: http://127.0.0.1:{args.port}", flush=True)
    print(f"Read-only: {summary['episode_count']} episodes, "
          f"{summary['total_frames']} recorded frames", flush=True)
    print("Remote access: SSH port forwarding only (no public network listener).",
          flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
