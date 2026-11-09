"""Interactive dark-mode Matplotlib dashboard for SFT vs SFT+RECAP.

Displays two matched LIBERO-10 Task 0 robot rollouts, numerical SIM-4 V(s)
curves *computed on those exact evaluation observations*, and optional SIM-5
training-only advantage details. V(s) is NOT the advantage function.

First-time use (records 20 new paired rollouts, then opens GUI)::

    python -m toolkits.recap_libero_dashboard \
      --checkpoint logs/cfg_rl/.../checkpoints/global_step_3000 \
      --episodes 20 --prepare --score-values

Reopen existing visualization without evaluating again::

    python -m toolkits.recap_libero_dashboard --episodes 20

If a display server is unavailable (e.g. plain SSH), create a static overview::

    MPLBACKEND=Agg python -m toolkits.recap_libero_dashboard \
      --episodes 20 --save-summary summary.png

Main window: two robot videos, two real SIM-4 value curves, 20-episode outcomes.
Training labels appear only when clicking the separate "Training labels" button.
Keyboard: space play/pause, left/right seek, up/down switch evaluation episode.
Shorter rollout holds its final frame so the paired timelines remain aligned.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

DEFAULT_OUTPUT = Path("logs/cfg_rl/recap_sim7_value_comparison_20episodes")
DEFAULT_SIDECAR = Path(
    "/data1/Muttaqien/rlinf_data/recap_libero10_task0/"
    "libero10_task0_train/meta/advantages_libero10_task0_recap.parquet"
)
BG = "#080a0d"
PANEL = "#15181e"
INK = "#f3f5f7"
SECONDARY = "#9aa4b0"
GREEN = "#43d5a1"
RED = "#ed7882"
BLUE = "#6dafff"
GRID = "#333c48"
FPS = 3  # SIM-7 saved frames[::10], encoded at 3 FPS; not wall-clock speed.


def _load_manifest(output: Path, episodes: int, seed: int) -> dict:
    path = output / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. First generate paired rollouts using --prepare."
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("episodes", -1)) != episodes or int(config.get("seed", -1)) != seed:
        raise ValueError(
            f"Existing videos were recorded with episodes={config.get('episodes')} "
            f"and seed={config.get('seed')}, not episodes={episodes}, seed={seed}. "
            "Choose matching arguments or a new --output directory."
        )
    return config


def _load_results(output: Path, episodes: int, seed: int) -> list[dict]:
    _load_manifest(output, episodes, seed)
    csv_path = output / "paired_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}; first generate rollouts with --prepare")
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != episodes:
        raise ValueError(f"Expected {episodes} paired results, got {len(rows)}")
    parsed = []
    for number, row in enumerate(rows, start=1):
        ep = int(row["episode"])
        ep_seed = int(row["seed"])
        if ep != number or ep_seed != seed + number - 1:
            raise ValueError(f"Episode {number}: video pairing/seed mismatch")
        videos = []
        for kind in ("baseline", "recap"):
            rel = Path(row[f"{kind}_video"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe video path in CSV: {rel}")
            path = (output / rel).resolve()
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty {kind} video for episode {ep}: {path}")
            videos.append(path)
        b = int(row["baseline_success"])
        r = int(row["recap_success"])
        if b not in (0, 1) or r not in (0, 1):
            raise ValueError("Expected binary success flags in paired_results.csv")
        parsed.append({
            "episode": ep, "seed": ep_seed,
            "baseline_success": b, "recap_success": r,
            "baseline_video": videos[0], "recap_video": videos[1],
        })
    return parsed


def _prepare_videos(output: Path, args: argparse.Namespace) -> None:
    if (output / "paired_results.csv").is_file():
        _load_results(output, args.episodes, args.seed)
        print(f"Reusing {args.episodes} existing paired episodes in {output}", flush=True)
        return
    if args.checkpoint is None:
        raise ValueError("--prepare needs --checkpoint when recordings do not already exist")
    command = [
        sys.executable, "-m", "toolkits.recap_libero_compare_videos",
        "--checkpoint", str(args.checkpoint.expanduser().resolve()),
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--output", str(output),
        "--model-path", str(args.model_path.expanduser().resolve()),
        "--value-observations",
    ]
    if args.reference_baseline is not None:
        command.extend(["--reference-baseline", str(args.reference_baseline),
                        "--reference-recap", str(args.reference_recap),
                        "--reference-trials", str(args.reference_trials)])
    print(f"Recording {args.episodes} paired episodes. No training is performed.", flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


def _read_video(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        frames = [np.asarray(frame, dtype=np.uint8).copy() for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No video frames in {path}")
    return frames


def _load_advantages(path: Path):
    """SIM-5 TRAINING sidecar, including V(s) and continuous A when available."""
    if not path.is_file():
        print(f"SIM-5 training sidecar missing: {path}; optional detail window unavailable.", flush=True)
        return None, None
    import pandas as pd
    mandatory = ["episode_index", "frame_index", "advantage"]
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(path).schema.names)
        columns = mandatory + [k for k in ("value_current", "advantage_continuous") if k in available]
        if not set(mandatory).issubset(available):
            raise ValueError(f"SIM-5 sidecar missing required columns: {set(mandatory)-available}")
        df = pd.read_parquet(path, columns=columns)
    elif path.suffix.lower() == ".csv":
        available = set(pd.read_csv(path, nrows=0).columns)
        if not set(mandatory).issubset(available):
            raise ValueError(f"SIM-5 sidecar missing required columns: {set(mandatory)-available}")
        columns = mandatory + [k for k in ("value_current", "advantage_continuous") if k in available]
        df = pd.read_csv(path, usecols=columns)
    else:
        raise ValueError("SIM-5 advantage sidecar must be .parquet or .csv")
    if df.empty or df[mandatory].isna().any().any():
        raise ValueError("SIM-5 sidecar contains null labels or no rows")
    if not df["advantage"].isin([0, 1, False, True]).all():
        raise ValueError("Expected binary SIM-5 training advantage labels")
    for col in ("value_current", "advantage_continuous"):
        if col in df and not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"SIM-5 {col} has nonfinite values")
    df = df.copy()
    df["advantage"] = df["advantage"].astype(np.int8)
    df["episode_index"] = df["episode_index"].astype(int)
    df["frame_index"] = df["frame_index"].astype(int)
    if df.duplicated(["episode_index", "frame_index"]).any():
        raise ValueError("SIM-5 sidecar has duplicate episode/frame pairs")
    df.sort_values(["episode_index", "frame_index"], inplace=True)
    episodes = df["episode_index"].drop_duplicates().to_numpy(dtype=int)
    print(f"Loaded SIM-5 training sidecar: {len(df):,} frames and {len(episodes)} episodes", flush=True)
    return df, episodes


def _read_curve(path: Path, episode: int, seed: int, policy: str, expected_frames: int):
    """Never substitute a fake curve when real numeric SIM-4 scores are absent."""
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_frames:
        raise ValueError(f"Value count mismatch in {path}: {len(rows)} vs {expected_frames} frames")
    numbers = []
    for i, row in enumerate(rows):
        if (int(row["episode"]) != episode or int(row["seed"]) != seed or
                row["policy"] != policy or int(row["frame_index"]) != i or
                int(row["sim_step"]) != 10 + i * 10):
            raise ValueError(f"Mismatched value metadata in {path}, row {i}")
        numbers.append(float(row["value"]))
    result = np.asarray(numbers, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite value estimates in {path}")
    return result


def _set_axis_style(ax, grid: bool = False) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    if grid:
        ax.grid(True, alpha=0.18, color=INK, linewidth=0.7)


class Dashboard:
    """Near-black Matplotlib video player with real SIM-4 evaluation value curves."""

    def __init__(self, rows, advantages, train_episodes, output, reference=None):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button, Slider

        self.plt = plt
        self.rows, self.output, self.reference = rows, output, reference
        self.advantages, self.train_episodes = advantages, train_episodes
        self.episode, self.frame_idx, self.train_idx = 0, 0, 0
        self.baseline, self.recap = [], []
        self.value_b, self.value_r = None, None
        self.playing, self._updating_progress, self.adv_fig = False, False, None
        self.fig = plt.figure(figsize=(15.8, 9.8), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title("RECAP | Paired robot and value-function comparison")
        except AttributeError:
            pass

        self.fig.text(.055, .965, "LIBERO-10  /  SFT vs. SFT + RECAP",
                      fontsize=19, color=INK, weight="bold", va="top")
        self.subtitle = self.fig.text(.055, .927, "", color=SECONDARY, fontsize=10.5)
        self.fig.text(.945, .965, "PI0.5  ·  RECAP  ·  SIM-7", ha="right",
                      va="top", fontsize=10.5, color=SECONDARY)

        # Two actual recorded robot cameras; no rendered placeholder during real runs.
        self.ax_b = self.fig.add_axes([.055, .482, .432, .373], facecolor=PANEL)
        self.ax_r = self.fig.add_axes([.513, .482, .432, .373], facecolor=PANEL)
        for ax in (self.ax_b, self.ax_r):
            _set_axis_style(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        self.artist_b = self.ax_b.imshow(blank, interpolation="nearest")
        self.artist_r = self.ax_r.imshow(blank, interpolation="nearest")
        self.ax_b.set_title("SFT BASELINE     /     no guidance", color=INK,
                            fontsize=13, weight="bold", pad=9)
        self.ax_r.set_title("SFT + RECAP     /     positive CFG", color=INK,
                            fontsize=13, weight="bold", pad=9)
        self.status_b = self.ax_b.text(.035, .045, "", transform=self.ax_b.transAxes,
                                      fontsize=12, fontweight="bold", color=GREEN,
                                      bbox=dict(facecolor=BG, alpha=.92, edgecolor=GRID, pad=6))
        self.status_r = self.ax_r.text(.035, .045, "", transform=self.ax_r.transAxes,
                                      fontsize=12, fontweight="bold", color=GREEN,
                                      bbox=dict(facecolor=BG, alpha=.92, edgecolor=GRID, pad=6))

        self.ax_vb = self.fig.add_axes([.067, .322, .403, .12], facecolor=PANEL)
        self.ax_vr = self.fig.add_axes([.525, .322, .403, .12], facecolor=PANEL)
        for ax in (self.ax_vb, self.ax_vr):
            _set_axis_style(ax, grid=True)
        self.value_axes = [self.ax_vb, self.ax_vr]
        self.value_lines = [None, None]
        self.cursors = [None, None]
        self.value_dots = [None, None]
        self.value_labels = [None, None]

        self.summary = self.fig.text(.055, .271, "", fontsize=11, color=INK, weight="bold")
        self.fig.text(.055, .229, "PAIRED OUTCOMES    •    Green = success    Red = failure",
                      color=SECONDARY, fontsize=9)
        self.ax_outcomes = self.fig.add_axes([.129, .163, .814, .057], facecolor=PANEL)
        _set_axis_style(self.ax_outcomes)
        outcome = np.array([[int(r["baseline_success"]) for r in rows],
                            [int(r["recap_success"]) for r in rows]])
        self.ax_outcomes.imshow(outcome, aspect="auto", interpolation="nearest",
                                cmap=ListedColormap([RED, GREEN]), vmin=0, vmax=1,
                                extent=(.5, len(rows) + .5, 1.5, -.5))
        self.ax_outcomes.set_yticks([0, 1], ["SFT", "RECAP"])
        self.ax_outcomes.tick_params(axis="y", length=0, pad=9)
        self.ax_outcomes.set_xticks(list(range(1, len(rows) + 1)) if len(rows) <= 25
                                    else np.linspace(1, len(rows), 10, dtype=int))
        self.ax_outcomes.tick_params(axis="x", length=0)
        self.ax_outcomes.set_xlim(.5, len(rows) + .5)
        self.ax_outcomes.set_ylim(1.5, -.5)
        self.outline = Rectangle((.5, -.5), 1, 2, fill=False,
                                 edgecolor=BLUE, linewidth=2.6)
        self.ax_outcomes.add_patch(self.outline)

        self.prev = Button(self.fig.add_axes([.055, .09, .09, .043]), "< Previous", color=PANEL, hovercolor=GRID)
        self.play = Button(self.fig.add_axes([.153, .09, .084, .043]), "Play", color=PANEL, hovercolor=GRID)
        self.next = Button(self.fig.add_axes([.245, .09, .09, .043]), "Next >", color=PANEL, hovercolor=GRID)
        self.labels_button = Button(self.fig.add_axes([.79, .09, .153, .043]),
                                    "SIM-5 details", color=PANEL, hovercolor=GRID)
        for b in (self.prev, self.play, self.next, self.labels_button):
            b.label.set_color(INK)
            b.label.set_fontsize(10)
        self.prev.on_clicked(lambda _: self.switch_episode(-1))
        self.play.on_clicked(lambda _: self.toggle_play())
        self.next.on_clicked(lambda _: self.switch_episode(1))
        self.labels_button.on_clicked(lambda _: self.show_advantages())
        slider_ax = self.fig.add_axes([.422, .105, .34, .016], facecolor=BG)
        self.progress = Slider(slider_ax, "Frame", 0, 1000, valinit=0,
                               valstep=1, color=BLUE)
        self.progress.label.set_color(SECONDARY)
        self.progress.valtext.set_color(SECONDARY)
        self.progress.on_changed(self._seek)
        self.footer = self.fig.text(.055, .055, "", color=SECONDARY, fontsize=9)
        self.fig.text(.055, .028,
                      "Robot: matched initial states  •  Value: SIM-4 critic V(s), not advantage or success probability  •  Sampled steps",
                      color=SECONDARY, fontsize=8.6)
        self.timer = self.fig.canvas.new_timer(interval=round(1000/3))
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda _: self.timer.stop())
        self.switch_episode(0)

    def _draw_value_axes(self):
        """Keep the SAME vertical scale for both policies, even if one ends early."""
        curves = (self.value_b, self.value_r)
        good = [x for x in curves if x is not None]
        if good:
            joined = np.concatenate(good)
            y0 = min(-1.0, float(joined.min()) - .03)
            y1 = max(0.0, float(joined.max()) + .03)
            if y1 - y0 < .05:
                y0, y1 = y0 - .03, y1 + .03
        else:
            y0, y1 = -1.03, .03
        for i, (ax, data, name) in enumerate(zip(
            self.value_axes, curves, ("SFT baseline", "SFT + RECAP"))):
            ax.clear()
            _set_axis_style(ax, grid=True)
            ax.set_title(f"VALUE ESTIMATE  V(s)  /  {name}", loc="left",
                         color=INK, fontsize=10.2, pad=7)
            ax.set_ylabel("Value", fontsize=8.5, color=SECONDARY)
            ax.set_ylim(y0, y1)
            frame_count = len(self.baseline if i == 0 else self.recap)
            ax.set_xlim(0, max(1, frame_count - 1))
            if data is not None:
                (self.value_lines[i],) = ax.plot(np.arange(len(data)), data,
                                color=INK, linewidth=1.85, zorder=3)
                self.value_dots[i], = ax.plot([], [], "o", color=BLUE, markersize=6, zorder=5)
                self.value_labels[i] = ax.text(.985, .95, "", transform=ax.transAxes,
                            ha="right", va="top", color=GREEN, fontsize=10, weight="bold",
                            bbox=dict(facecolor=PANEL, alpha=.85, edgecolor="none"))
            else:
                self.value_lines[i], self.value_dots[i], self.value_labels[i] = None, None, None
                ax.text(.5, .52, "V(s) not yet scored", transform=ax.transAxes,
                        ha="center", va="center", color=SECONDARY, fontsize=10)
                ax.text(.5, .23, "Run --score-values with SIM-4 checkpoint",
                        transform=ax.transAxes, ha="center", va="center",
                        color=SECONDARY, fontsize=8)
            self.cursors[i] = ax.axvline(0, color=BLUE, linewidth=1.2,
                                         linestyle="--", alpha=.95)

    def _update_frame(self):
        if not self.baseline or not self.recap:
            return
        i = self.frame_idx
        self.artist_b.set_data(self.baseline[min(i, len(self.baseline)-1)])
        self.artist_r.set_data(self.recap[min(i, len(self.recap)-1)])
        row = self.rows[self.episode]
        for status, success, ended in (
            (self.status_b, row["baseline_success"], i >= len(self.baseline)),
            (self.status_r, row["recap_success"], i >= len(self.recap)),
        ):
            status.set_text(("SUCCESS" if success else "FAIL") + ("  /  ENDED" if ended else ""))
            status.set_color(GREEN if success else RED)
        for data, pointer, dot, label in zip(
            (self.value_b, self.value_r), self.cursors,
            self.value_dots, self.value_labels,
        ):
            if pointer is not None:
                effective = i if data is None else min(i, len(data)-1)
                pointer.set_xdata([effective, effective])
                if data is not None:
                    dot.set_data([effective], [data[effective]])
                    label.set_text(f"V = {data[effective]:+.3f}")
        count = max(len(self.baseline), len(self.recap))
        self.subtitle.set_text(
            f"LIBERO-10 Task 0   |   Episode {row['episode']:02d}/{len(self.rows)}   |   "
            f"Matched seed {row['seed']}   |   Video frame {i+1}/{count}"
        )
        self.outline.set_x(self.episode+.5)
        self.fig.canvas.draw_idle()

    def _set_frame(self, i):
        count = max(len(self.baseline), len(self.recap))
        self.frame_idx = max(0, min(int(i), count-1))
        self._updating_progress = True
        try:
            self.progress.set_val(round(1000*self.frame_idx/max(1, count-1)))
        finally:
            self._updating_progress = False
        self._update_frame()

    def _seek(self, value):
        if self._updating_progress or not self.baseline or not self.recap:
            return
        count = max(len(self.baseline), len(self.recap))
        self.frame_idx = round(float(value) * max(0, count-1) / 1000)
        self._update_frame()

    def switch_episode(self, change):
        index = max(0, min(self.episode + change, len(self.rows)-1))
        if self.baseline and index == self.episode:
            return
        if self.playing:
            self.toggle_play()
        self.episode = index
        row = self.rows[self.episode]
        print(f"Loading paired episode {row['episode']}/{len(self.rows)} (seed {row['seed']})", flush=True)
        self.baseline = _read_video(row["baseline_video"])
        self.recap = _read_video(row["recap_video"])
        self.value_b = _read_curve(self.output / "values" / f"baseline_episode{row['episode']:02d}.csv",
                                   row["episode"], row["seed"], "baseline", len(self.baseline))
        self.value_r = _read_curve(self.output / "values" / f"recap_episode{row['episode']:02d}.csv",
                                   row["episode"], row["seed"], "recap", len(self.recap))
        self._draw_value_axes()
        self._set_frame(0)
        b = sum(x["baseline_success"] for x in self.rows)
        r = sum(x["recap_success"] for x in self.rows)
        self.summary.set_text(
            f"20-episode comparison" if len(self.rows) == 20 else f"{len(self.rows)}-episode comparison"
        )
        self.summary.set_text(self.summary.get_text() +
                              f"     SFT {b}/{len(self.rows)} ({b/len(self.rows):.0%})"
                              f"     |     RECAP {r}/{len(self.rows)} ({r/len(self.rows):.0%})")
        if self.reference is not None:
            rb, rr, rt = self.reference
            self.footer.set_text(
                f"Previous {rt}-episode benchmark (separate sample): SFT {rb}/{rt} vs RECAP {rr}/{rt}"
            )
        else:
            self.footer.set_text("Results on this screen are from the paired recorded sample only.")
        self.fig.canvas.draw_idle()

    def toggle_play(self):
        if not self.baseline or not self.recap:
            return
        self.playing = not self.playing
        self.play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            if self.frame_idx >= max(len(self.baseline), len(self.recap))-1:
                self._set_frame(0)
            self.timer.start()
        else:
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def _tick(self):
        if not self.playing:
            return
        count = max(len(self.baseline), len(self.recap))
        if self.frame_idx >= count-1:
            self.toggle_play()
        else:
            self._set_frame(self.frame_idx+1)

    def show_advantages(self):
        """Separate popup for SIM-5 *training* V(s) and A_cont, not eval labels."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        if self.adv_fig is not None and plt.fignum_exists(self.adv_fig.number):
            self._draw_advantages()
            self.adv_fig.canvas.manager.show()
            return
        self.adv_fig = plt.figure(figsize=(10.8, 7.1), facecolor=BG)
        try:
            self.adv_fig.canvas.manager.set_window_title("RECAP | SIM-5 training value and advantage")
        except AttributeError:
            pass
        self.adv_fig.text(.07, .96, "SIM-5 training signals", color=INK,
                          fontsize=17, weight="bold", va="top")
        self.adv_fig.text(.07, .905,
            "Training trajectories only. These are NOT the advantage estimates of the evaluation videos.",
            color=SECONDARY, fontsize=9.8)
        self.adv_info = self.adv_fig.text(.07, .86, "", color=INK, fontsize=10.5)
        self.ax_train_v = self.adv_fig.add_axes([.095, .51, .83, .29], facecolor=PANEL)
        self.ax_train_a = self.adv_fig.add_axes([.095, .17, .83, .26], facecolor=PANEL)
        self.adv_prev = Button(self.adv_fig.add_axes([.10, .045, .17, .057]),
                               "< Train episode", color=PANEL, hovercolor=GRID)
        self.adv_next = Button(self.adv_fig.add_axes([.29, .045, .17, .057]),
                               "Train episode >", color=PANEL, hovercolor=GRID)
        for button in (self.adv_prev, self.adv_next):
            button.label.set_color(INK)
        self.adv_prev.on_clicked(lambda _: self.switch_training_episode(-1))
        self.adv_next.on_clicked(lambda _: self.switch_training_episode(1))
        self.adv_fig.canvas.mpl_connect("key_press_event", self._on_adv_key)
        self.adv_fig.canvas.mpl_connect("close_event", lambda _: setattr(self, "adv_fig", None))
        self._draw_advantages()
        self.adv_fig.show()

    def switch_training_episode(self, change):
        if self.train_episodes is None or len(self.train_episodes) == 0:
            return
        self.train_idx = max(0, min(self.train_idx + change, len(self.train_episodes)-1))
        self._draw_advantages()

    def _draw_advantages(self):
        if self.adv_fig is None:
            return
        ax_v, ax_a = self.ax_train_v, self.ax_train_a
        for ax in (ax_v, ax_a):
            ax.clear()
            _set_axis_style(ax, grid=True)
        if self.advantages is None or self.train_episodes is None or len(self.train_episodes)==0:
            self.adv_info.set_text("SIM-5 training sidecar unavailable")
            ax_v.text(.5, .5, "No SIM-5 training signals", color=SECONDARY,
                      transform=ax_v.transAxes, ha="center")
            self.adv_fig.canvas.draw_idle()
            return
        ep = int(self.train_episodes[self.train_idx])
        subset = self.advantages.loc[self.advantages["episode_index"] == ep]
        frames = subset["frame_index"].to_numpy(dtype=int)
        labels = subset["advantage"].to_numpy(dtype=int)
        if "value_current" in subset:
            ax_v.plot(frames, subset["value_current"].to_numpy(dtype=float),
                      color=INK, linewidth=1.8)
            ax_v.set_title("V(s)  /  SIM-4 value critic on SIM-5 training observations",
                           color=INK, loc="left", fontsize=11)
        else:
            ax_v.text(.5,.5,"No saved numerical V(s) in SIM-5 sidecar",transform=ax_v.transAxes,
                      ha="center",color=SECONDARY)
        ax_v.set_ylabel("Value", color=SECONDARY)
        if "advantage_continuous" in subset:
            advantage = subset["advantage_continuous"].to_numpy(dtype=float)
            ax_a.plot(frames, advantage, color=BLUE, linewidth=1.6)
            ax_a.axhline(0, color=SECONDARY, linestyle="--", linewidth=.9)
            ax_a.set_title("A(s,a)  /  SIM-5 numerical n-step advantage estimate",
                           color=INK, loc="left", fontsize=11)
            ax_a.set_ylabel("Advantage", color=SECONDARY)
        else:
            ax_a.scatter(frames[labels == 1], labels[labels == 1],
                         color=GREEN, s=9, label="positive")
            ax_a.scatter(frames[labels == 0], labels[labels == 0],
                         color=RED, s=9, label="negative")
            ax_a.set_title("SIM-5 training advantage labels (binary)",
                           color=INK, loc="left", fontsize=11)
        ax_a.set_xlabel("Training frame", color=SECONDARY)
        self.adv_info.set_text(
            f"Training episode {self.train_idx+1}/{len(self.train_episodes)} (ID {ep})    "
            f"{len(frames)} frames    {labels.mean():.1%} positive labels"
        )
        self.adv_fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == " ":
            self.toggle_play()
        elif event.key == "left":
            self._set_frame(self.frame_idx-1)
        elif event.key == "right":
            self._set_frame(self.frame_idx+1)
        elif event.key == "up":
            self.switch_episode(-1)
        elif event.key == "down":
            self.switch_episode(1)
        elif event.key in ("a", "A"):
            self.show_advantages()

    def _on_adv_key(self, event):
        if event.key in ("left", "a", "A"):
            self.switch_training_episode(-1)
        elif event.key in ("right", "d", "D"):
            self.switch_training_episode(1)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="RECAP global_step_3000 directory; needed for --prepare")
    parser.add_argument("--model-path", type=Path, default=Path("/data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_libero_sft_rlinf"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare", action="store_true", help="Record synchronized front/wrist/state LIBERO rollouts (no retraining)")
    parser.add_argument("--score-values", action="store_true", help="Compute actual V(s) for evaluation recordings using SIM-4 critic")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Record/score videos without opening the GUI (safe over plain SSH)")
    parser.add_argument("--value-checkpoint", type=Path, default=None, help="Override SIM-4 value critic checkpoint")
    parser.add_argument("--value-batch-size", type=int, default=4)
    parser.add_argument("--advantage-parquet", type=Path, default=DEFAULT_SIDECAR,
                        help="SIM-5 training advantage labels (.parquet or .csv); never applied to eval videos")
    parser.add_argument("--reference-baseline", type=int, default=None)
    parser.add_argument("--reference-recap", type=int, default=None)
    parser.add_argument("--reference-trials", type=int, default=None)
    parser.add_argument("--save-summary", type=Path, default=None,
                        help="Save a static PNG overview instead of opening a desktop GUI")
    args = parser.parse_args()
    if args.episodes < 1 or args.value_batch_size < 1:
        parser.error("--episodes and --value-batch-size must be >= 1")
    if args.prepare_only and not (args.prepare or args.score_values):
        parser.error("--prepare-only needs --prepare and/or --score-values")
    ref_tuple = (args.reference_baseline, args.reference_recap, args.reference_trials)
    if any(x is not None for x in ref_tuple):
        if not all(x is not None for x in ref_tuple):
            parser.error("Specify all three --reference-* arguments")
        if not (0 <= ref_tuple[0] <= ref_tuple[2] and 0 <= ref_tuple[1] <= ref_tuple[2] and ref_tuple[2] > 0):
            parser.error("Reference counts must satisfy 0 <= successes <= trials")
        reference = ref_tuple
    else:
        reference = None
    import matplotlib
    if args.save_summary is not None:
        matplotlib.use("Agg")
    elif not args.prepare_only and matplotlib.get_backend().lower() == "agg":
        parser.error("No interactive Matplotlib display detected. To record/score "
                     "from SSH, pass --prepare-only; to save a preview, "
                     "pass --save-summary path.png; to view videos interactively, "
                     "run on a graphical desktop or with X11 forwarding.")
    output = args.output.expanduser().resolve()
    if args.prepare:
        _prepare_videos(output, args)
    rows = _load_results(output, args.episodes, args.seed)
    if args.score_values:
        from toolkits.recap_libero_value_curves import DEFAULT_CHECKPOINT, score
        score(output, args.value_checkpoint or DEFAULT_CHECKPOINT,
              args.episodes, args.seed, args.value_batch_size)
    if args.prepare_only:
        print(f"Prepared {len(rows)} paired episodes at {output}. "
              "Open the GUI later from the graphical desktop.", flush=True)
        return
    adv, train_episodes = _load_advantages(args.advantage_parquet.expanduser().resolve())
    import matplotlib.pyplot as plt
    dashboard = Dashboard(rows, adv, train_episodes, output, reference)
    if args.save_summary is not None:
        saved = args.save_summary.expanduser().resolve()
        saved.parent.mkdir(parents=True, exist_ok=True)
        dashboard.fig.savefig(saved, dpi=150, facecolor=BG)
        print(f"Saved dark-mode overview: {saved}", flush=True)
        plt.close(dashboard.fig)
    else:
        print("GUI ready. Space play/pause; up/down episode; click SIM-5 details for training signals.", flush=True)
        plt.show()


if __name__ == "__main__":
    main()
