"""Record and visualize paired LIBERO-10 Task 0 SFT vs SFT+RECAP rollouts.

Runs the existing SIM-7 evaluator twice using identical initial states and seeds,
then creates synchronized side-by-side MP4s plus a local HTML gallery.

Examples:
  python -m toolkits.recap_libero_compare_videos \
    --checkpoint logs/cfg_rl/.../checkpoints/global_step_3000 \
    --episodes 5 --reference-baseline 7 --reference-recap 19 --reference-trials 30

  # Build comparison videos again without rerunning robot rollouts:
  python -m toolkits.recap_libero_compare_videos \
    --compose-only --output logs/cfg_rl/recap_sim7_video_comparison \
    --episodes 5

The per-episode status is read from the SIM-7 evaluator's actual MP4 filenames.
Video frames are aligned by sampled simulator step. When one rollout ends earlier,
its last frame is held until the other finishes; this is NOT real-time playback.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import os
import shutil
import zipfile
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_SFT = "/data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_libero_sft_rlinf"
VIDEO_FPS = 3  # The existing SIM-7 evaluator writes frames[::10] at 3 fps.
STEPS_PER_FRAME = 10
WARMUP_STEPS = 10
PANEL_SIZE = 448
GAP = 16
HEADER = 96
FOOTER = 64
BACKGROUND = (18, 23, 34)
WHITE = (244, 247, 251)
MUTED = (169, 181, 200)
SUCCESS = (68, 196, 142)
FAILURE = (245, 106, 112)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _episode_video(folder: Path, episode: int) -> tuple[Path, bool]:
    candidates = list(folder.glob(f"task0_episode{episode}_*.mp4"))
    exact = [
        path for path in candidates
        if re.fullmatch(rf"task0_episode{episode}_(success|failure)\.mp4", path.name)
    ]
    if len(exact) != 1:
        raise RuntimeError(
            f"Expected exactly one video for episode {episode} in {folder}, "
            f"found {len(exact)}. Avoid mixing runs in the same directory."
        )
    return exact[0], exact[0].stem.endswith("_success")


def _frames(video: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(video), format="ffmpeg")
    try:
        frames = [np.asarray(frame).copy() for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"Video contains no frames: {video}")
    return frames


def _fit_frame(frame: np.ndarray, size: int) -> Image.Image:
    rgb = Image.fromarray(frame).convert("RGB")
    fitted = ImageOps.contain(rgb, (size, size), method=Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), (35, 42, 55))
    panel.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return panel


def _draw_frame(
    baseline: Image.Image,
    recap: Image.Image,
    baseline_ok: bool,
    recap_ok: bool,
    episode: int,
    seed: int,
    step: int,
    baseline_ended: bool,
    recap_ended: bool,
) -> np.ndarray:
    width = PANEL_SIZE * 2 + GAP * 3
    height = HEADER + PANEL_SIZE + FOOTER
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    x_left = GAP
    x_right = PANEL_SIZE + GAP * 2
    image_y = HEADER
    canvas.paste(baseline, (x_left, image_y))
    canvas.paste(recap, (x_right, image_y))

    draw.text((x_left, 12), "SFT baseline", font=_font(25), fill=WHITE)
    draw.text((x_right, 12), "SFT + RECAP", font=_font(25), fill=WHITE)
    draw.text((x_left, 45), "pi0.5 LIBERO-SFT | no guidance", font=_font(16), fill=MUTED)
    draw.text((x_right, 45), "pi0.5 LIBERO-SFT | positive CFG", font=_font(16), fill=MUTED)
    for x, succeeded, ended in [
        (x_left, baseline_ok, baseline_ended),
        (x_right, recap_ok, recap_ended),
    ]:
        color = SUCCESS if succeeded else FAILURE
        status = "SUCCESS" if succeeded else "FAIL"
        draw.rounded_rectangle((x, 70, x + 93, 92), radius=6, fill=color)
        draw.text((x + 8, 71), status, font=_font(15), fill=(10, 20, 25))
        if ended:
            draw.text((x + 109, 72), "ROLLOUT ENDED - final frame held", font=_font(13), fill=MUTED)

    draw.text((GAP, HEADER + PANEL_SIZE + 12),
              f"LIBERO-10 task 0  |  Episode {episode}  |  Seed {seed}  |  Sampled step ~{step}",
              font=_font(17), fill=WHITE)
    draw.text((GAP, HEADER + PANEL_SIZE + 38),
              "Same initial state and seed; frames aligned by simulator step, not wall-clock time.",
              font=_font(13), fill=MUTED)
    return np.asarray(canvas)


def _compose_episode(
    baseline_file: Path,
    recap_file: Path,
    baseline_ok: bool,
    recap_ok: bool,
    episode: int,
    seed: int,
    output: Path,
    montage_writer,
) -> int:
    baseline_frames = _frames(baseline_file)
    recap_frames = _frames(recap_file)
    count = max(len(baseline_frames), len(recap_frames))
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output), fps=VIDEO_FPS) as writer:
        for index in range(count):
            b_ended = index >= len(baseline_frames)
            r_ended = index >= len(recap_frames)
            b = _fit_frame(baseline_frames[min(index, len(baseline_frames) - 1)], PANEL_SIZE)
            r = _fit_frame(recap_frames[min(index, len(recap_frames) - 1)], PANEL_SIZE)
            frame = _draw_frame(
                b, r, baseline_ok, recap_ok, episode, seed,
                WARMUP_STEPS + index * STEPS_PER_FRAME, b_ended, r_ended,
            )
            writer.append_data(frame)
            montage_writer.append_data(frame)
    return count


def _run_evaluator(
    repo_root: Path,
    model_path: Path,
    checkpoint: Path | None,
    output: Path,
    episodes: int,
    seed: int,
    value_observations: bool = False,
    episode_indices: list[int] | None = None,
) -> None:
    command = [
        sys.executable, "-m", "toolkits.recap_libero_sim7_eval",
        "--model-path", str(model_path),
        "--trials", str(episodes),
        "--video-count", str(episodes),
        "--seed", str(seed),
        "--output", str(output.resolve()),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint.resolve())])
    if value_observations:
        command.append("--value-observations")
    if episode_indices:
        command.extend(["--episode-indices", ",".join(map(str, episode_indices))])
    print(f"Running {'SFT + RECAP' if checkpoint else 'SFT baseline'} evaluation...", flush=True)
    subprocess.run(command, cwd=repo_root, check=True)



def _matching_files(folder: Path, episode: int) -> list[Path]:
    """Return all episode-specific artifacts, including interrupted/partial outputs."""
    return sorted(
        list(folder.glob(f"task0_episode{episode}_*.mp4"))
        + list(folder.glob(f"wrist_task0_episode{episode}_*.mp4"))
        + ([folder / f"state_task0_episode{episode}.npz"]
           if (folder / f"state_task0_episode{episode}.npz").exists() else [])
    )


def _valid_existing_episode(folder: Path, episode: int, seed: int, value_observations: bool) -> bool:
    """Only reuse COMPLETE recordings; never infer success from a partial MP4."""
    files = _matching_files(folder, episode)
    if not files:
        return False
    front = [p for p in files
             if re.fullmatch(rf"task0_episode{episode}_(success|failure)\.mp4", p.name)]
    if len(front) != 1 or not front[0].is_file():
        return False
    try:
        baseline = _frames(front[0])
        if not value_observations:
            return bool(baseline)
        suffix = "success" if front[0].stem.endswith("_success") else "failure"
        wrist = folder / f"wrist_task0_episode{episode}_{suffix}.mp4"
        state = folder / f"state_task0_episode{episode}.npz"
        if not wrist.is_file() or not state.is_file():
            return False
        wrist_frames = _frames(wrist)
        with np.load(state, allow_pickle=False) as item:
            states = np.asarray(item["states"])
            saved_seed = int(item["seed"])
            saved_episode = int(item["episode"])
            prompt = str(item["prompt"].item())
        return (
            saved_seed == seed and saved_episode == episode and bool(prompt)
            and len(baseline) == len(wrist_frames) == len(states)
            and states.ndim == 2 and states.shape[1] == 8
            and np.isfinite(states).all()
            and len(files) == 3
        )
    except (OSError, ValueError, KeyError, RuntimeError, IndexError, EOFError, zipfile.BadZipFile):
        return False


def _recover_policy(
    folder: Path, policy: str, episodes: int, seed: int, value_observations: bool,
    backup_root: Path,
) -> list[int]:
    """Preserve interrupted artifacts, then return only missing episode numbers."""
    missing = []
    for episode in range(1, episodes + 1):
        if _valid_existing_episode(folder, episode, seed + episode - 1, value_observations):
            print(f"Reusing complete {policy} episode {episode}/{episodes}", flush=True)
            continue
        leftovers = _matching_files(folder, episode)
        if leftovers:
            dest = backup_root / policy
            dest.mkdir(parents=True, exist_ok=True)
            for item in leftovers:
                shutil.move(str(item), str(dest / item.name))
            print(f"Preserved {len(leftovers)} partial {policy} episode {episode} file(s) in {dest}", flush=True)
        missing.append(episode)
    return missing



def _write_gallery(output: Path, rows: list[dict], reference: tuple[int, int, int] | None):
    n = len(rows)
    b = sum(row["baseline_success"] for row in rows)
    r = sum(row["recap_success"] for row in rows)
    reference_html = ""
    if reference:
        base, recap, trials = reference
        reference_html = (
            f"<section class='results'><h2>Previous {trials}-episode benchmark</h2>"
            f"<p>SFT baseline: <strong>{base}/{trials} ({base / trials:.1%})</strong>"
            f" &nbsp; vs &nbsp; SFT + RECAP: <strong>{recap}/{trials} ({recap / trials:.1%})</strong>.</p>"
            "<p class='note'>Historical results supplied by the user; these are NOT calculated from the paired video sample.</p>"
            "</section>"
        )
    videos = "\n".join(
        f"<section class='episode'><h2>Episode {row['episode']} &mdash; "
        f"SFT {'success' if row['baseline_success'] else 'failure'} / "
        f"RECAP {'success' if row['recap_success'] else 'failure'}</h2>"
        f"<video controls preload='metadata' playsinline src='{html.escape(row['comparison_video'], quote=True)}'></video>"
        f"<p class='note'>Same initial state, episode index and seed ({row['seed']}); shorter rollout freezes at its final frame.</p>"
        "</section>"
        for row in rows
    )
    document = f"""<!DOCTYPE html>
<html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>RECAP / LIBERO policy video comparison</title>
<style>
:root {{ color-scheme: dark; }}
body {{ margin:0 auto; padding:24px; max-width:1100px; background:#101620; color:#f3f7fc; font-family:system-ui,sans-serif; }}
h1 {{ margin-bottom:8px; }}
.note {{ color:#aab9ce; font-size:14px; }}
.results, .episode {{ background:#1b2534; border:1px solid #354258; border-radius:14px; padding:18px; margin:18px 0; }}
h2 {{ font-size:19px; }}
video {{ display:block; width:100%; height:auto; border-radius:8px; }}
strong {{ color:#68deb1; }}
</style></head><body>
<h1>LIBERO-10 Task 0: SFT vs SFT + RECAP</h1>
<p>Matched initial states and seeds, synchronized by sampled simulator step.</p>
<section class='results'><h2>Paired video sample ({n} episodes)</h2>
<p>SFT baseline: <strong>{b}/{n} ({b/n:.1%})</strong> &nbsp; vs &nbsp;
SFT + RECAP: <strong>{r}/{n} ({r/n:.1%})</strong>.</p>
<p class='note'>These are only the episodes included in this gallery; do not treat them as a 30-episode estimate.</p></section>
{reference_html}
{videos}
</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="RECAP global_step_3000 directory")
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_SFT))
    parser.add_argument("--episodes", type=int, default=5, help="Number of paired rollouts (default 5)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("logs/cfg_rl/recap_sim7_video_comparison"))
    parser.add_argument("--compose-only", action="store_true", help="Reuse existing baseline/recap MP4s")
    parser.add_argument("--resume", action="store_true", help="Recover complete existing episodes; preserve incomplete recordings and rerun only missing episodes")
    parser.add_argument("--value-observations", action="store_true",
                        help="Record wrist views and states for optional numerical SIM-4 V(s) scoring")
    parser.add_argument("--reference-baseline", type=int, help="Optional historical baseline success count")
    parser.add_argument("--reference-recap", type=int, help="Optional historical RECAP success count")
    parser.add_argument("--reference-trials", type=int, help="Optional historical total episode count")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    ref_args = (args.reference_baseline, args.reference_recap, args.reference_trials)
    if any(v is not None for v in ref_args) and not all(v is not None for v in ref_args):
        parser.error("Specify all three reference counts together")
    if all(v is not None for v in ref_args) and not (0 <= ref_args[0] <= ref_args[2] and 0 <= ref_args[1] <= ref_args[2] and ref_args[2] > 0):
        parser.error("Reference counts must satisfy 0 <= success <= trials")
    reference = tuple(ref_args) if all(v is not None for v in ref_args) else None

    repo_root = Path(__file__).resolve().parents[1]
    output = args.output.expanduser().resolve()
    baseline_dir = output / "baseline"
    recap_dir = output / "recap"
    if args.compose_only and args.resume:
        parser.error("Choose --compose-only or --resume, not both")
    manifest_path = output / "run_config.json"
    if not args.compose_only:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --compose-only is set")
        checkpoint = args.checkpoint.expanduser().resolve()
        if not (checkpoint / "actor/model_state_dict/full_weights.pt").is_file():
            parser.error(f"No RECAP weights found in {checkpoint}")
        model_path = args.model_path.expanduser().resolve()
        if not model_path.is_dir():
            parser.error(f"SFT model directory missing: {model_path}")
        manifest = {
            "seed": args.seed, "episodes": args.episodes,
            "model_path": str(model_path), "checkpoint": str(checkpoint),
            "value_observations": args.value_observations,
        }
        if manifest_path.is_file():
            original = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(original.get(key) != value for key, value in manifest.items()):
                parser.error("Existing run_config.json does not match this experiment. Use a new --output.")
        else:
            if any(baseline_dir.glob("*.mp4")) or any(recap_dir.glob("*.mp4")):
                if not args.resume:
                    parser.error(f"Existing rollout MP4s found in {output}; use --resume to inspect and recover them, or a new --output")
                print("WARNING: Existing recordings have no run manifest. The episode seeds and video completeness can be validated, but the original model checkpoint cannot be proven from those files. --resume explicitly accepts this limitation.", flush=True)
            output.mkdir(parents=True, exist_ok=True)
            pending_manifest = manifest_path.with_suffix(".json.tmp")
            pending_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            pending_manifest.replace(manifest_path)
        if not args.resume and (any(baseline_dir.glob("*.mp4")) or any(recap_dir.glob("*.mp4"))):
            parser.error(f"Existing rollout MP4s found in {output}; use --resume or a new --output")
        if args.resume:
            backup_root = output / "interrupted_backup" / (datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
            missing_baseline = _recover_policy(
                baseline_dir, "baseline", args.episodes, args.seed, args.value_observations, backup_root,
            )
            missing_recap = _recover_policy(
                recap_dir, "recap", args.episodes, args.seed, args.value_observations, backup_root,
            )
        else:
            missing_baseline = list(range(1, args.episodes + 1))
            missing_recap = list(range(1, args.episodes + 1))
        if missing_baseline:
            _run_evaluator(repo_root, model_path, None, baseline_dir, args.episodes, args.seed, args.value_observations, missing_baseline)
        if missing_recap:
            _run_evaluator(repo_root, model_path, checkpoint, recap_dir, args.episodes, args.seed, args.value_observations, missing_recap)
        # Verify every episode before composing outcome files. Do not report incomplete jobs as success.
        for folder, policy in ((baseline_dir, "baseline"), (recap_dir, "recap")):
            for episode in range(1, args.episodes + 1):
                if not _valid_existing_episode(folder, episode, args.seed + episode - 1, args.value_observations):
                    raise RuntimeError(f"Recorded {policy} episode {episode} is incomplete; do not compose misleading results")
    else:
        if manifest_path.is_file():
            original = json.loads(manifest_path.read_text(encoding="utf-8"))
            if original.get("seed") != args.seed or original.get("episodes") != args.episodes:
                parser.error("--seed/--episodes differ from run_config.json; use recorded settings")
        else:
            parser.error("--compose-only requires run_config.json to verify matching seeds and episodes")

    rows = []
    montage = output / "comparison_all_episodes.mp4"
    with imageio.get_writer(str(montage), fps=VIDEO_FPS) as montage_writer:
        for episode in range(1, args.episodes + 1):
            baseline_file, baseline_ok = _episode_video(baseline_dir, episode)
            recap_file, recap_ok = _episode_video(recap_dir, episode)
            paired_name = f"comparison_episode{episode:02d}.mp4"
            count = _compose_episode(
                baseline_file, recap_file, baseline_ok, recap_ok,
                episode, args.seed + episode - 1, output / paired_name, montage_writer,
            )
            rows.append({
                "episode": episode,
                "seed": args.seed + episode - 1,
                "baseline_success": int(baseline_ok),
                "recap_success": int(recap_ok),
                "baseline_video": str(baseline_file.relative_to(output)),
                "recap_video": str(recap_file.relative_to(output)),
                "comparison_video": paired_name,
            })
            print(f"Paired episode {episode}: SFT={'SUCCESS' if baseline_ok else 'FAIL'}; "
                  f"RECAP={'SUCCESS' if recap_ok else 'FAIL'}; {count} frames", flush=True)

    with (output / "paired_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_gallery(output, rows, reference)
    print(f"\nComparison video: {montage}")
    print(f"Interactive gallery: {output / 'index.html'}")
    print(f"Paired results:     {output / 'paired_results.csv'}")


if __name__ == "__main__":
    main()
