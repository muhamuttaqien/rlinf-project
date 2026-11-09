#!/usr/bin/env python3
"""Local Matplotlib GUI for recorded FR3 episodes, SIM-theme color palette.

Runs inside the branch-openpi-recap-franka Git worktree without replacing the
original dataset-side fr3_visualize.py:

    python -m toolkits.realworld_check.fr3_local_visualizer \
        --root /data1/Muttaqien/fr3_peg_100

Supports --episode, --speed and --save-overviews like the original.
Identical 3x2 camera/plot layout and interactions, with the SIM RECAP
dashboard's charcoal, blue, green and red design tokens.

Requirements: numpy and matplotlib. No ROS, robot connection or camera SDK.
Raw recordings remain read-only; --save-overviews writes output PNG files
inside <root>/visualizations/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping invalid JSON {path.name}:{line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


def vector(state: dict[str, Any], name: str, length: int) -> np.ndarray:
    try:
        values = np.asarray(state.get(name, []), dtype=float).reshape(-1)
        if len(values) < length:
            return np.full(length, np.nan)
        return values[:length]
    except (TypeError, ValueError):
        return np.full(length, np.nan)


def finite_extent(data: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    mn, mx = float(values.min()), float(values.max())
    if mn == mx:
        eps = max(0.01, abs(mn) * 0.02)
        return mn - eps, mx + eps
    margin = (mx - mn) * 0.08
    return mn - margin, mx + margin


def event_magnitude(event: dict[str, Any]) -> float:
    msg = event.get("message") or {}
    if not isinstance(msg, dict):
        return math.nan
    axes = []
    for key in ("linear", "angular"):
        d = msg.get(key, {})
        if not isinstance(d, dict):
            return math.nan
        try:
            axes.extend(float(d[k]) for k in ("x", "y", "z"))
        except (TypeError, ValueError, KeyError):
            return math.nan
    return float(np.linalg.norm(axes))


@dataclass
class Episode:
    path: Path
    records: list[dict[str, Any]]
    commands: list[dict[str, Any]]
    manifest: dict[str, Any]
    review: dict[str, Any]
    times: np.ndarray
    joint: np.ndarray
    tcp: np.ndarray
    grip: np.ndarray
    force: np.ndarray
    twist_times: np.ndarray
    twist_values: np.ndarray
    grip_cmd_times: np.ndarray
    grip_cmd_values: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "Episode":
        records = read_jsonl(path / "states.jsonl")
        if not records:
            raise ValueError(f"No saved state samples in {path}")
        records.sort(key=lambda r: int(r.get("frame_index", 0)))
        commands = read_jsonl(path / "operator_commands.jsonl")
        manifest = read_json(path / "manifest.json")
        review = read_json(path / "episode_review.json")
        start = records[0].get("recorded_monotonic_ns")
        if not isinstance(start, (int, float)):
            start = records[0].get("recorded_unix_ns", 0)
            time_key = "recorded_unix_ns"
            event_key = "received_unix_ns"
        else:
            time_key = "recorded_monotonic_ns"
            event_key = "received_monotonic_ns"

        times = np.array([float(r.get(time_key, start) - start) * 1e-9 for r in records])
        states = [r.get("measured_state") or {} for r in records]
        joint = np.array([vector(s, "arm_joint_position", 7) for s in states])
        tcp = np.array([vector(s, "tcp_pose", 7) for s in states])
        grip = np.array([vector(s, "robotiq_knuckle_position", 1)[0] for s in states])
        force = np.array([vector(s, "tcp_force", 3) for s in states])

        twist = []
        grip_cmd = []
        for event in commands:
            ts = event.get(event_key)
            if not isinstance(ts, (int, float)):
                continue
            t = (float(ts) - float(start)) * 1e-9
            if event.get("name") == "operator_twist":
                twist.append((t, event_magnitude(event)))
            elif event.get("name") == "operator_gripper_request":
                try:
                    grip_cmd.append((t, float(event.get("message", {}).get("data"))))
                except (ValueError, TypeError, AttributeError):
                    pass
        return cls(
            path, records, commands, manifest, review, times, joint, tcp,
            grip, force,
            np.array([v[0] for v in twist]), np.array([v[1] for v in twist]),
            np.array([v[0] for v in grip_cmd]), np.array([v[1] for v in grip_cmd]),
        )

    def image(self, index: int, role: str) -> np.ndarray:
        rec = self.records[index]
        files = rec.get("image_files_bgr8") or {}
        if not isinstance(files, dict):
            files = {}
        rel = files.get(role)
        if not rel:
            idx = int(rec.get("frame_index", index))
            rel = f"frames/{role}/{idx:06d}.npy"
        path = (self.path / rel).resolve()
        # Only read files inside the selected episode.
        if self.path.resolve() not in path.parents:
            raise ValueError(f"Unexpected image path: {rel}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Invalid BGR frame shape: {arr.shape} ({path})")
        # Files are BGR8, while Matplotlib expects RGB8.
        return np.ascontiguousarray(arr[:, :, ::-1])


def episode_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("episode_*" ) if p.is_dir() and (p / "states.jsonl").is_file())


# Shared palette with toolkits/recap_libero_web_dashboard.html (SIM).
# Retain legacy CYAN/PINK names for compatibility with the original GUI code.
BG = "#07090c"
PANEL = "#10141a"
PANEL_ALT = "#1d2633"
GRID = "#27313e"
FG = "#edf3f8"
MUTED = "#94a1b4"
CYAN = "#75aafb"
BLUE = "#75aafb"
GREEN = "#58d9b0"
RED = "#fb8a91"
PURPLE = "#c4aff5"
AMBER = "#efc783"
PINK = RED
PRIMARY = "#3478c7"
PRIMARY_HOVER = "#438ad8"
TRACE_COLORS = [BLUE, GREEN, RED, PURPLE, AMBER, "#9ec6ef", "#a3d0b1"]
SPEEDS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def decorate_plot(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, loc="left", color=FG, size=10, weight="bold", pad=13)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=7)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)


def plot_lines(ax, times, data, names, ylabel, title):
    decorate_plot(ax, title, ylabel)
    for i, name in enumerate(names):
        y = data[:, i] if data.ndim > 1 else data
        ax.plot(times, y, lw=1.5, label=name,
                color=TRACE_COLORS[i % len(TRACE_COLORS)], alpha=0.94)
    extent = finite_extent(data)
    if extent:
        ax.set_ylim(*extent)
    ax.legend(loc="upper right", fontsize=7.7, frameon=False,
              labelcolor=MUTED, ncol=min(len(names), 4),
              borderaxespad=0.3, handlelength=1.3, columnspacing=0.7)


class Viewer:
    """Offline, time-synchronized preview. 1x means one recorded second per real second.

    Use recorded monotonic timestamps rather than assuming a fixed camera FPS.
    When the GUI cannot redraw every image, skip display frames to stay on time.
    The underlying recordings are never altered.
    """

    def __init__(self, root: Path, paths: list[Path], initial: int,
                 show: bool, speed: float = 1.0):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.plt = plt
        self.paths = paths
        self.position = initial
        self.data = Episode.load(self.paths[self.position])
        self.index = 0
        self.playing = False
        self.speed = float(speed)
        self._clock_origin = time.perf_counter()
        self._timeline_origin = 0.0
        self._internal_slider_update = False
        self._loading = False
        self._warning_keys = set()
        self._image_cache = OrderedDict()  # limited-size RGB cache
        self._max_cache_pairs = 8

        plt.rcParams["font.family"] = "DejaVu Sans"
        self.fig = plt.figure(figsize=(16.5, 9.8), facecolor=BG)
        self.fig.canvas.manager.set_window_title("FR3  |  Offline Data Review") if hasattr(
            self.fig.canvas.manager, "set_window_title") else None
        grid = self.fig.add_gridspec(
            3, 2, left=0.055, right=0.945, bottom=0.232, top=0.830,
            height_ratios=[2.65, 1.08, 1.08], hspace=0.64, wspace=0.18,
        )
        self.ax_base = self.fig.add_subplot(grid[0, 0])
        self.ax_wrist = self.fig.add_subplot(grid[0, 1])
        self.ax_tcp = self.fig.add_subplot(grid[1, 0])
        self.ax_joint = self.fig.add_subplot(grid[1, 1])
        self.ax_grip = self.fig.add_subplot(grid[2, 0])
        self.ax_cmd = self.fig.add_subplot(grid[2, 1])
        self.curves = (self.ax_tcp, self.ax_joint, self.ax_grip, self.ax_cmd)
        self.rgb = {}
        self.cursors = []

        self.fig.text(0.055, 0.963, "F R 3   /   D A T A   R E V I E W",
                      fontsize=10, color=CYAN, weight="bold", va="center")
        self.fig.text(0.945, 0.963, "OFFLINE  •  READ ONLY",
                      fontsize=10, color=MUTED, ha="right", va="center")
        self.heading = self.fig.text(0.055, 0.918, "", fontsize=23,
                                     color=FG, weight="bold", va="center")
        self.description = self.fig.text(0.055, 0.877, "", fontsize=10,
                                         color=MUTED, va="center")
        self.progress_text = self.fig.text(0.055, 0.185, "", fontsize=10,
                                           color=FG, va="center")
        self.telemetry_text = self.fig.text(0.945, 0.185, "", fontsize=10,
                                            color=MUTED, ha="right", va="center")
        self.fig.text(0.055, 0.019,
                      "SPACE  play/pause     ← →  frame     PgUp / PgDn  episode     ↑ ↓  speed     HOME / END  jump",
                      fontsize=8.7, color=MUTED)

        self.ax_slider = self.fig.add_axes([0.10, 0.139, 0.81, 0.024], facecolor=BG)
        self.slider = Slider(
            self.ax_slider, "FRAME", 0, max(1, len(self.data.records) - 1),
            valinit=0, valstep=1, color=CYAN, track_color=GRID,
            handle_style={"facecolor": FG, "edgecolor": CYAN, "size": 7},
        )
        self.slider.label.set_color(MUTED)
        self.slider.label.set_fontsize(9)
        self.slider.valtext.set_visible(False)
        self.slider.on_changed(self._on_slider)

        self.prev = self._button(Button, [0.055, 0.060, 0.15, 0.052], "‹  PREV EPISODE")
        self.next = self._button(Button, [0.215, 0.060, 0.15, 0.052], "NEXT EPISODE  ›")
        self.back = self._button(Button, [0.410, 0.060, 0.065, 0.052], "−1F")
        self.play = self._button(Button, [0.485, 0.060, 0.128, 0.052], "▶  PLAY", primary=True)
        self.forward = self._button(Button, [0.623, 0.060, 0.065, 0.052], "+1F")
        self.slower = self._button(Button, [0.756, 0.060, 0.050, 0.052], "−")
        self.speed_btn = self._button(Button, [0.814, 0.060, 0.075, 0.052], "1.0×")
        self.faster = self._button(Button, [0.896, 0.060, 0.050, 0.052], "+")
        self.prev.on_clicked(lambda _: self.switch(-1))
        self.next.on_clicked(lambda _: self.switch(+1))
        self.back.on_clicked(lambda _: self.seek(self.index - 1, user=True))
        self.forward.on_clicked(lambda _: self.seek(self.index + 1, user=True))
        self.play.on_clicked(lambda _: self.toggle())
        self.slower.on_clicked(lambda _: self.adjust_speed(-1))
        self.faster.on_clicked(lambda _: self.adjust_speed(+1))
        self.speed_btn.on_clicked(lambda _: self.set_speed(1.0))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", lambda _: self.timer.stop())
        self.timer = self.fig.canvas.new_timer(interval=30)
        self.timer.add_callback(self.tick)
        self.build_episode()
        if show:
            plt.show()

    def _button(self, cls, xywh, label, primary=False):
        ax = self.fig.add_axes(xywh)
        item = cls(ax, label, color=PRIMARY if primary else PANEL_ALT,
                   hovercolor=PRIMARY_HOVER if primary else GRID)
        item.label.set_color(FG)
        item.label.set_weight("bold")
        item.label.set_fontsize(9.5)
        return item

    def _cached_image(self, index, role):
        key = (index, role)
        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            return self._image_cache[key]
        image = self.data.image(index, role)
        self._image_cache[key] = image
        if len(self._image_cache) > 2 * self._max_cache_pairs:
            self._image_cache.popitem(last=False)
        return image

    def _image_or_placeholder(self, index, role):
        try:
            return self._cached_image(index, role)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            key = (self.data.path.name, role, index)
            if key not in self._warning_keys:
                print(f"WARNING: missing {role} frame {index}: {exc}", file=sys.stderr)
                self._warning_keys.add(key)
            return np.full((240, 400, 3), [23, 36, 56], dtype=np.uint8)

    def build_episode(self):
        self._loading = True
        self._image_cache.clear()
        d = self.data
        for ax in (self.ax_base, self.ax_wrist, *self.curves):
            ax.clear()
        self.rgb.clear()
        self.cursors.clear()

        for role, ax, title in (
            ("base", self.ax_base, "CAM 01   /   BASE VIEW"),
            ("wrist", self.ax_wrist, "CAM 02   /   WRIST VIEW"),
        ):
            ax.set_facecolor(PANEL)
            self.rgb[role] = ax.imshow(self._image_or_placeholder(0, role),
                                       interpolation="nearest", aspect="equal")
            ax.set_title(title, loc="left", color=FG, size=10, weight="bold", pad=13)
            ax.axis("off")

        plot_lines(self.ax_tcp, d.times, d.tcp[:, :3],
                   ["X", "Y", "Z"], "metres", "TCP  /  CARTESIAN POSITION")
        plot_lines(self.ax_joint, d.times, d.joint,
                   [f"J{i}" for i in range(1, 8)], "radians", "FR3  /  JOINT ANGLES")
        plot_lines(self.ax_grip, d.times, d.grip,
                   ["measured"], "radians", "ROBOTIQ  /  KNUCKLE ANGLE")

        decorate_plot(self.ax_cmd, "SPACEMOUSE  /  OPERATOR INPUT (NOT EXECUTED)",
                      "normalized")
        if len(d.twist_times):
            self.ax_cmd.plot(d.twist_times, d.twist_values, ".", markersize=2.3,
                             color=CYAN, alpha=0.78, label="Twist magnitude")
        if len(d.grip_cmd_times):
            self.ax_cmd.step(d.grip_cmd_times, d.grip_cmd_values, where="post",
                             lw=1.5, color=AMBER, label="Gripper request")
            self.ax_cmd.scatter(d.grip_cmd_times, d.grip_cmd_values, s=16,
                                color=AMBER, zorder=4)
        if len(d.twist_times) or len(d.grip_cmd_times):
            self.ax_cmd.legend(loc="upper right", fontsize=8, frameon=False,
                               labelcolor=MUTED)
        else:
            self.ax_cmd.text(0.5, 0.5, "No recorded operator events",
                             ha="center", va="center", transform=self.ax_cmd.transAxes,
                             color=MUTED, fontsize=10)
        extent = finite_extent(np.r_[d.twist_values, d.grip_cmd_values])
        if extent:
            self.ax_cmd.set_ylim(*extent)

        for ax in self.curves:
            ax.set_xlim(min(0.0, float(d.times.min())), max(0.1, float(d.times.max())))
            ax.set_xlabel("Time (s)", fontsize=8, color=MUTED)
            self.cursors.append(ax.axvline(float(d.times[0]), color=FG,
                                              linestyle="--", linewidth=1.2,
                                              alpha=0.72, zorder=10))

        outcome = str(d.review.get("operator_outcome", "unreviewed"))
        status = str(d.review.get("capture_status", d.manifest.get("status", "unknown")))
        task = str(d.manifest.get("task", d.review.get("task", "FR3 recording")))
        duration = float(d.times[-1] - d.times[0])
        self.heading.set_text(d.path.name.replace("_", "  ").upper())
        self.description.set_text(
            f"{task}    •    {outcome.upper()}  /  {status.upper()}    •    "
            f"{len(d.records)} frames    •    {duration:.2f} s    •    "
            f"{len(d.commands)} input events    •    episode {self.position+1}/{len(self.paths)}"
        )
        self.slider.valmax = max(1, len(d.records) - 1)
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self._internal_slider_update = True
        self.slider.set_val(0)
        self._internal_slider_update = False
        self.index = 0
        self._clock_origin = time.perf_counter()
        self._timeline_origin = float(d.times[0])
        self._loading = False
        self._refresh_frame()
        self._refresh_buttons()

    def _refresh_buttons(self):
        self.play.label.set_text("Ⅱ  PAUSE" if self.playing else "▶  PLAY")
        self.speed_btn.label.set_text(f"{self.speed:g}×")
        self.fig.canvas.draw_idle()

    def _refresh_frame(self):
        d = self.data
        for role in ("base", "wrist"):
            self.rgb[role].set_data(self._image_or_placeholder(self.index, role))
        t = float(d.times[self.index])
        for cursor in self.cursors:
            cursor.set_xdata([t, t])
        force = d.force[self.index]
        fm = float(np.linalg.norm(force)) if np.all(np.isfinite(force)) else math.nan
        self.progress_text.set_text(
            f"FRAME  {self.index + 1:04d} / {len(d.records):04d}      "
            f"{t:.2f} / {float(d.times[-1]):.2f} s      "
            f"PLAYBACK  {self.speed:g}×"
        )
        self.telemetry_text.set_text(f"Measured TCP force  {fm:.2f} N")
        self.fig.canvas.draw_idle()

    def _on_slider(self, value):
        if not self._internal_slider_update and not self._loading:
            self.seek(int(value), user=True)

    def seek(self, index, user=False):
        idx = max(0, min(int(index), len(self.data.records) - 1))
        self.index = idx
        if user:
            # If seeking while playing, continue at the selected timestamp.
            self._timeline_origin = float(self.data.times[idx])
            self._clock_origin = time.perf_counter()
        if int(self.slider.val) != idx:
            self._internal_slider_update = True
            self.slider.set_val(idx)
            self._internal_slider_update = False
        self._refresh_frame()

    def tick(self):
        if not self.playing:
            return
        d = self.data
        target = self._timeline_origin + (time.perf_counter() - self._clock_origin) * self.speed
        if target >= float(d.times[-1]):
            self.seek(len(d.records) - 1)
            self.playing = False
            self.timer.stop()
            self._refresh_buttons()
            return
        # Frame skipping is intentional: match original elapsed time at 1x even
        # when two images + a full Matplotlib dashboard draw take >100 ms.
        idx = max(0, int(np.searchsorted(d.times, target, side="right") - 1))
        if idx != self.index:
            self.seek(idx)

    def toggle(self):
        if self.playing:
            self.playing = False
            self.timer.stop()
        else:
            if self.index >= len(self.data.records) - 1:
                self.seek(0)
            self._timeline_origin = float(self.data.times[self.index])
            self._clock_origin = time.perf_counter()
            self.playing = True
            self.timer.start()
        self._refresh_buttons()

    def set_speed(self, speed):
        if speed <= 0 or not math.isfinite(speed):
            return
        if self.playing:
            self._timeline_origin = self._timeline_origin + (
                time.perf_counter() - self._clock_origin) * self.speed
            self._clock_origin = time.perf_counter()
        self.speed = float(speed)
        self._refresh_frame()
        self._refresh_buttons()

    def adjust_speed(self, step):
        pos = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - self.speed))
        self.set_speed(SPEEDS[max(0, min(len(SPEEDS) - 1, pos + step))])

    def switch(self, delta):
        if self.playing:
            self.playing = False
            self.timer.stop()
        self.position = (self.position + delta) % len(self.paths)
        self.data = Episode.load(self.paths[self.position])
        self.build_episode()

    def on_key(self, event):
        if event.key in (" ", "space"):
            self.toggle()
        elif event.key == "right":
            self.seek(self.index + 1, user=True)
        elif event.key == "left":
            self.seek(self.index - 1, user=True)
        elif event.key == "up":
            self.adjust_speed(+1)
        elif event.key == "down":
            self.adjust_speed(-1)
        elif event.key == "pagedown":
            self.switch(1)
        elif event.key == "pageup":
            self.switch(-1)
        elif event.key == "home":
            self.seek(0, user=True)
        elif event.key == "end":
            self.seek(len(self.data.records) - 1, user=True)


def choose_index(paths: list[Path], value: str | None) -> int:
    if not value:
        return 0
    desired = value if value.startswith("episode_") else f"episode_{int(value):04d}"
    for i, path in enumerate(paths):
        if path.name == desired:
            return i
    raise ValueError(f"Episode not found: {desired}. Available: {', '.join(p.name for p in paths)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data1/Muttaqien/fr3_peg_100"),
                        help="FR3 raw episode dataset root")
    parser.add_argument("--episode", default=None, help="Start episode, e.g. 2 or episode_0002")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Initial playback speed (default 1x, recorded-time accurate)")
    parser.add_argument("--save-overviews", action="store_true",
                        help="Export one static PNG per episode to visualizations/ (no GUI)")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be a positive finite number")
    if args.save_overviews:
        import matplotlib
        matplotlib.use("Agg")
    root = args.root.expanduser().resolve()
    paths = episode_paths(root)
    if not paths:
        parser.error(f"No episode_*/states.jsonl files found under {root}")
    selected = choose_index(paths, args.episode)
    if args.save_overviews:
        import matplotlib.pyplot as plt
        outdir = root / "visualizations"
        outdir.mkdir(exist_ok=True)
        targets = [paths[selected]] if args.episode else paths
        for path in targets:
            viewer = Viewer(root, [path], 0, False, speed=args.speed)
            viewer.seek(len(viewer.data.records) // 2, user=True)
            output = outdir / f"{path.name}_overview.png"
            viewer.fig.savefig(output, dpi=125, bbox_inches="tight")
            plt.close(viewer.fig)
            print(f"Saved {output}")
        return 0
    print(f"Viewing {len(paths)} episodes from {root} (read-only)")
    Viewer(root, paths, selected, True, speed=args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())