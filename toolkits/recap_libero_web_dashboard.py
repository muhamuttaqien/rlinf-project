"""Serve a private browser dashboard for paired LIBERO SFT vs RECAP videos.

Read-only: serves previously recorded 20 paired episodes and optional SIM-4
value curves. It neither launches simulation nor trains/scorers models.

Lab workstation (terminal 1):
    python -m toolkits.recap_libero_web_dashboard --episodes 20 --port 8765

Windows Command Prompt (terminal 2, with lab VPN if needed):
    ssh -N -L 8765:127.0.0.1:8765 muttaqien@10.17.94.130

Then open http://127.0.0.1:8765 in the Windows browser.

IMPORTANT: The default bind is 127.0.0.1. Do not expose laboratory
experiment videos or their metadata on a publicly accessible interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "logs/cfg_rl/recap_sim7_value_comparison_20episodes"
DEFAULT_SIDECAR = Path(
    "/data1/Muttaqien/rlinf_data/recap_libero10_task0/"
    "libero10_task0_train/meta/advantages_libero10_task0_recap.parquet"
)
HTML_FILE = Path(__file__).with_suffix(".html")
SUITES = (
    ("libero_spatial", "LIBERO-Spatial", "Spatial relationships"),
    ("libero_object", "LIBERO-Object", "Object variation"),
    ("libero_goal", "LIBERO-Goal", "Goal variation"),
    ("libero_10", "LIBERO-Long (LIBERO-10)", "Long-horizon manipulation"),
    ("libero_90", "LIBERO-90", "Diverse short-horizon tasks"),
)
SUITE_IDS = frozenset(item[0] for item in SUITES)
MEDIA_RE = re.compile(r"^/media/(?:(libero_spatial|libero_object|libero_goal|libero_10|libero_90)/)?(baseline|recap)/(\d{1,4})$")
VIDEO_NAME_RE = re.compile(r"^task0_episode(\d+)_(success|failure)\.mp4$")
MAX_API_POINTS = 5000


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def load_experiment(output: Path, *, expected_episodes: int | None = 20, expected_seed: int | None = None, expected_suite: str = "libero_10", expected_task_index: int = 0) -> tuple[dict, list[dict]]:
    """Validate the exact experiment and video paths before serving anything."""
    output = output.expanduser().resolve()
    manifest_path = output / "run_config.json"
    results_path = output / "paired_results.csv"
    if not (manifest_path.is_file() and results_path.is_file()):
        raise FileNotFoundError(
            f"Paired experiment not found in {output}. First run the existing "
            "Matplotlib dashboard with --prepare --score-values --prepare-only."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = int(manifest.get("episodes", -1))
    seed = int(manifest.get("seed", -1))
    if manifest.get("suite", "libero_10") != expected_suite or int(manifest.get("task_index", 0)) != expected_task_index:
        raise ValueError(f"Recorded task identity mismatch for {output}: expected {expected_suite} task {expected_task_index}")
    if episodes < 1 or episodes > 1000 or seed < 0:
        raise ValueError("Invalid episode count or seed in run_config.json")
    if expected_episodes is not None and episodes != expected_episodes:
        raise ValueError(f"Recorded experiment has {episodes} episodes, not {expected_episodes}")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(f"Recorded experiment has seed {seed}, not {expected_seed}")

    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != episodes:
        raise ValueError(f"Expected {episodes} paired_results.csv entries, found {len(rows)}")

    validated = []
    for episode, row in enumerate(rows, start=1):
        ep, ep_seed = int(row["episode"]), int(row["seed"])
        if ep != episode or ep_seed != seed + episode - 1:
            raise ValueError(f"Episode {episode} index/seed mismatch")
        record = {"episode": ep, "seed": ep_seed}
        for policy in ("baseline", "recap"):
            success = int(row[f"{policy}_success"])
            if success not in (0, 1):
                raise ValueError(f"Episode {episode} has invalid {policy} success")
            rel = Path(row[f"{policy}_video"])
            expected_parent = policy
            matched = VIDEO_NAME_RE.fullmatch(rel.name)
            if (rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 2
                or rel.parts[0] != expected_parent or not matched
                or int(matched.group(1)) != episode
                or matched.group(2) != ("success" if success else "failure")):
                raise ValueError(f"Unsafe or mismatched {policy} video path for episode {episode}: {rel}")
            video = output / rel
            if not video.is_file() or video.stat().st_size == 0 or not _inside(video, output):
                raise FileNotFoundError(f"Missing or invalid {policy} video: {video}")
            record[f"{policy}_success"] = success
            record[f"{policy}_video"] = video.resolve()
            record[f"{policy}_value"] = _value_curve_path(output, policy, episode)
        validated.append(record)
    return manifest, validated


def _value_curve_path(output: Path, policy: str, episode: int) -> Path | None:
    path = output / "values" / f"{policy}_episode{episode:02d}.csv"
    if path.is_file() and _inside(path, output):
        return path.resolve()
    return None


def read_value_curve(path: Path | None, episode: int, seed: int, policy: str) -> list[float] | None:
    """Use only validated numerical SIM-4 results; never synthesize a curve."""
    if path is None:
        return None
    result = []
    with path.open(newline="", encoding="utf-8") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            if (int(row["episode"]) != episode or int(row["seed"]) != seed
                or row["policy"] != policy or int(row["frame_index"]) != i
                or int(row["sim_step"]) != 10 + i * 10):
                raise ValueError(f"SIM-4 curve metadata mismatch in {path}, row {i}")
            val = float(row["value"])
            if not math.isfinite(val):
                raise ValueError(f"Nonfinite SIM-4 value estimate in {path}, row {i}")
            result.append(val)
    if not result or len(result) > MAX_API_POINTS:
        raise ValueError(f"Invalid SIM-4 value series length in {path}: {len(result)}")
    return result


class TrainingSidecar:
    """Lazy, optional cache of SIM-5 TRAINING labels, never evaluation values."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self._lock = threading.Lock()
        self._episodes = None
        self._error = None
        self._data = None
        self._count = 0

    def _load(self) -> None:
        if self._data is not None or self._error is not None:
            return
        with self._lock:
            if self._data is not None or self._error is not None:
                return
            if not self.path.is_file():
                self._error = f"SIM-5 training sidecar not found: {self.path}"
                return
            try:
                import pandas as pd  # available in the training environment
                cols = ["episode_index", "frame_index", "advantage", "value_current", "advantage_continuous"]
                if self.path.suffix == ".parquet":
                    import pyarrow.parquet as pq  # likely present in LeRobot env
                    available = set(pq.ParquetFile(str(self.path)).schema.names)
                    names = [name for name in cols if name in available]
                    df = pd.read_parquet(self.path, columns=names)
                elif self.path.suffix == ".csv":
                    available = set(pd.read_csv(self.path, nrows=0).columns)
                    df = pd.read_csv(self.path, usecols=[name for name in cols if name in available])
                else:
                    raise ValueError("Expected .parquet or .csv SIM-5 sidecar")
                if not {"episode_index", "frame_index", "advantage"}.issubset(df.columns):
                    raise ValueError("SIM-5 sidecar must contain episode_index/frame_index/advantage")
                if df.empty or df[["episode_index", "frame_index", "advantage"]].isna().any().any():
                    raise ValueError("SIM-5 sidecar is empty or contains null labels")
                if not df["advantage"].isin([0, 1, True, False]).all():
                    raise ValueError("Nonbinary SIM-5 advantage labels; refusing to reinterpret them")
                for col in ("value_current", "advantage_continuous"):
                    if col in df.columns:
                        if not all(math.isfinite(float(x)) for x in df[col].to_numpy()):
                            raise ValueError(f"SIM-5 {col} has non-finite values")
                df = df.sort_values(["episode_index", "frame_index"]).copy()
                df["advantage"] = df["advantage"].astype(bool)
                self._episodes = sorted(int(x) for x in df["episode_index"].unique())
                self._data = df
                self._count = len(df)
            except (OSError, ValueError, ImportError, KeyError, TypeError) as exc:
                self._error = f"Cannot read SIM-5 sidecar: {exc}"

    def get(self, requested_episode: int | None) -> dict:
        self._load()
        if self._error:
            return {"available": False, "message": self._error}
        ids = self._episodes
        if not ids:
            return {"available": False, "message": "No SIM-5 training episodes found"}
        ep = ids[0] if requested_episode is None else requested_episode
        if ep not in ids:
            raise ValueError(f"Unknown training episode {ep}")
        subset = self._data.loc[self._data["episode_index"] == ep]
        if len(subset) > MAX_API_POINTS:
            # A full training trajectory can be larger than a browser chart.
            stride = math.ceil(len(subset) / MAX_API_POINTS)
            subset = subset.iloc[::stride]
        return {
            "available": True,
            "episode": ep,
            "episodes": ids,
            "total_frames": self._count,
            "frames": subset["frame_index"].astype(int).tolist(),
            "labels": subset["advantage"].astype(int).tolist(),
            "values": subset["value_current"].astype(float).tolist() if "value_current" in subset else None,
            "advantages": subset["advantage_continuous"].astype(float).tolist() if "advantage_continuous" in subset else None,
        }


class Experiment:
    def __init__(self, output: Path, expected_episodes: int, seed: int | None, training_sidecar: Path,
                 reference: tuple[int, int, int] | None, suite: str = "libero_10", task_index: int = 0):
        self.output = output.expanduser().resolve()
        self.manifest, self.rows = load_experiment(self.output, expected_episodes=expected_episodes, expected_seed=seed, expected_suite=suite, expected_task_index=task_index)
        self.suite = suite
        self.task_index = task_index
        self.sidecar = TrainingSidecar(training_sidecar)
        self.reference = reference

    def summary(self) -> dict:
        b = sum(r["baseline_success"] for r in self.rows)
        r = sum(r["recap_success"] for r in self.rows)
        return {
            "task": f"{self.suite} Task {self.task_index}",
            "suite": self.suite, "task_index": self.task_index,
            "episodes": len(self.rows), "fps": 3, "seed": int(self.manifest["seed"]),
            "baseline_successes": b, "recap_successes": r,
            "value_observations": bool(self.manifest.get("value_observations", False)),
            "reference": ({"baseline": self.reference[0], "recap": self.reference[1],
                           "trials": self.reference[2]} if self.reference else None),
            "rows": [
                {"episode": row["episode"], "seed": row["seed"],
                 "baseline_success": row["baseline_success"], "recap_success": row["recap_success"],
                 "baseline_has_value": row["baseline_value"] is not None,
                 "recap_has_value": row["recap_value"] is not None}
                for row in self.rows
            ],
        }

    def episode(self, number: int) -> dict:
        if number < 1 or number > len(self.rows):
            raise ValueError(f"Episode must be 1..{len(self.rows)}")
        row = self.rows[number - 1]
        return {
            "episode": number, "seed": row["seed"],
            "baseline": {"success": bool(row["baseline_success"]),
                         "url": f"/media/baseline/{number}",
                         "values": read_value_curve(row["baseline_value"], number, row["seed"], "baseline")},
            "recap": {"success": bool(row["recap_success"]),
                      "url": f"/media/recap/{number}",
                      "values": read_value_curve(row["recap_value"], number, row["seed"], "recap")},
        }



class SuiteCollection:
    """Five independent paired experiments. Never alias one suite to another."""

    def __init__(self, long_output: Path, episodes: int, seed: int | None,
                 sidecar: Path, reference: tuple[int, int, int] | None):
        self.episodes = episodes
        self.datasets: dict[str, Experiment | None] = {}
        self.errors: dict[str, str] = {}
        root = long_output.expanduser().resolve()
        for suite, _label, _focus in SUITES:
            folder = root if suite == "libero_10" else root.parent / f"{root.name}_{suite}"
            if not (folder / "run_config.json").is_file() or not (folder / "paired_results.csv").is_file():
                self.datasets[suite] = None
                continue
            try:
                self.datasets[suite] = Experiment(
                    folder, episodes, seed, sidecar,
                    reference if suite == "libero_10" else None,
                    suite=suite, task_index=0,
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self.datasets[suite] = None
                self.errors[suite] = str(exc)
        self.default_suite = "libero_10"

    def _get(self, suite: str) -> Experiment | None:
        if suite not in SUITE_IDS:
            raise ValueError(f"Unknown LIBERO suite: {suite}")
        return self.datasets[suite]

    def catalog(self) -> dict:
        rows = []
        for suite, name, focus in SUITES:
            experiment = self.datasets[suite]
            rows.append({
                "id": suite, "name": name, "focus": focus,
                "task_index": 0, "task": "Task 0", "episodes": self.episodes,
                "ready": experiment is not None,
                "status": "Invalid data" if suite in self.errors else (
                    "Ready" if experiment is not None else "Not recorded"
                ),
            })
        return {"suites": rows, "default_suite": self.default_suite}

    def summary(self, suite: str) -> dict:
        experiment = self._get(suite)
        name = next(label for key, label, _focus in SUITES if key == suite)
        if experiment is None:
            return {
                "suite": suite, "category": name, "task": "Task 0", "task_index": 0,
                "available": False, "episodes": self.episodes, "rows": [],
                "message": self.errors.get(suite, f"{name} Task 0 has not been recorded. Run the 20-paired-episode evaluation for this suite."),
            }
        return {**experiment.summary(), "available": True, "category": name,
                "task": "Task 0", "task_description": experiment.manifest.get("task_description"),
                "value_critic_scope": "SIM-4 critic trained on LIBERO-10 Task 0; other suites are out-of-distribution."}

    def episode(self, suite: str, number: int) -> dict:
        experiment = self._get(suite)
        if experiment is None:
            raise ValueError(f"Suite {suite} has no recorded paired episodes")
        episode = experiment.episode(number)
        for policy in ("baseline", "recap"):
            episode[policy]["url"] = f"/media/{suite}/{policy}/{number}"
        return episode

    def training(self, suite: str, number: int | None) -> dict:
        if suite != "libero_10":
            return {"available": False, "message": "SIM-5 training signals originate from LIBERO-10 Task 0 and do not apply to this suite."}
        experiment = self._get(suite)
        if experiment is None:
            return {"available": False, "message": "LIBERO-10 Task 0 is not recorded."}
        return experiment.sidecar.get(number)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RECAP-ReadOnly/1.0"

    def log_message(self, fmt, *args):
        # Suppress query strings and local file paths in the workstation log.
        return

    @property
    def experiment(self) -> Experiment:
        return self.server.experiment

    def _common_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; connect-src 'self'; media-src 'self'; "
                         "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")

    def _send(self, status: int, content: bytes, content_type: str = "application/json; charset=utf-8", head: bool = False):
        self.send_response(status)
        self._common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if not head:
            self.wfile.write(content)

    def _json(self, data: dict, status: int = 200, head: bool = False):
        self._send(status, json.dumps(data, allow_nan=False, separators=(",", ":")).encode("utf-8"), head=head)

    def _media(self, policy: str, number: int, *, head: bool = False, experiment: Experiment | None = None):
        experiment = experiment or self.experiment
        if number < 1 or number > len(experiment.rows):
            return self._json({"error": "Episode not found"}, 404, head)
        video = experiment.rows[number - 1][f"{policy}_video"]
        try:
            filesize = video.stat().st_size
        except OSError:
            return self._json({"error": "Video missing"}, 404, head)
        start, end = 0, filesize - 1
        range_header = self.headers.get("Range")
        is_partial = False
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or (not match[1] and not match[2]):
                return self._bad_range(filesize, head)
            first, last = match.groups()
            if first:
                start = int(first)
                if start >= filesize:
                    return self._bad_range(filesize, head)
                end = min(int(last), filesize - 1) if last else filesize - 1
                if end < start:
                    return self._bad_range(filesize, head)
            else:
                suffix = int(last)
                if suffix <= 0:
                    return self._bad_range(filesize, head)
                start = max(0, filesize - suffix)
            is_partial = True
        length = end - start + 1
        self.send_response(206 if is_partial else 200)
        self._common_headers()
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if is_partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{filesize}")
        self.end_headers()
        if head:
            return
        try:
            with video.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Browser aborts ranges on seek or episode switching; expected.
            pass

    def _bad_range(self, size: int, head: bool):
        self.send_response(416)
        self._common_headers()
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _handle(self, *, head: bool = False):
        url = urlsplit(self.path)
        path = url.path
        params = parse_qs(url.query)
        try:
            if path in ("/", "/index.html"):
                return self._send(200, HTML_FILE.read_bytes(), "text/html; charset=utf-8", head)
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon", head)
            collection = getattr(self.server, "collection", None)
            suite = params.get("suite", ["libero_10"])[0]
            if path == "/api/suites":
                return self._json(collection.catalog() if collection else {
                    "suites": [{"id": "libero_10", "name": "LIBERO-Long (LIBERO-10)", "task": "Task 0", "episodes": len(self.experiment.rows), "ready": True, "status": "Ready"}],
                    "default_suite": "libero_10",
                }, head=head)
            if path == "/api/summary":
                return self._json(collection.summary(suite) if collection else self.experiment.summary(), head=head)
            if path == "/api/episode":
                number = int(params.get("episode", ["1"])[0])
                return self._json(collection.episode(suite, number) if collection else self.experiment.episode(number), head=head)
            if path == "/api/training":
                selection = params.get("episode", [None])[0]
                requested = None if selection is None else int(selection)
                return self._json(collection.training(suite, requested) if collection else self.experiment.sidecar.get(requested), head=head)
            match = MEDIA_RE.fullmatch(path)
            if match:
                chosen = match[1] or "libero_10"
                experiment = collection._get(chosen) if collection else self.experiment
                if experiment is None:
                    return self._json({"error": "Suite has no recorded videos"}, 404, head)
                return self._media(match[2], int(match[3]), head=head, experiment=experiment)
            return self._json({"error": "Not found"}, 404, head)
        except (ValueError, TypeError, KeyError) as exc:
            return self._json({"error": str(exc)}, 400, head)
        except (OSError, json.JSONDecodeError) as exc:
            return self._json({"error": "Cannot read experiment data", "detail": type(exc).__name__}, 500, head)

    def do_GET(self):
        self._handle()

    def do_HEAD(self):
        self._handle(head=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Existing LIBERO-10 Task 0 directory; the other suites are discovered in sibling folders suffixed with their suite IDs")
    parser.add_argument("--episodes", type=int, default=20, help="Expected paired episode count")
    parser.add_argument("--seed", type=int, default=None, help="Optional expected seed; inferred from manifest")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR, help="Optional SIM-5 training advantage sidecar")
    parser.add_argument("--reference-baseline", type=int, default=None)
    parser.add_argument("--reference-recap", type=int, default=None)
    parser.add_argument("--reference-trials", type=int, default=None)
    args = parser.parse_args()
    refs = (args.reference_baseline, args.reference_recap, args.reference_trials)
    if any(x is not None for x in refs):
        if not all(x is not None for x in refs) or not (0 <= refs[0] <= refs[2] and 0 <= refs[1] <= refs[2] and refs[2] > 0):
            parser.error("Reference results require all three values, 0 <= successes <= trials")
        reference = refs
    else:
        reference = None
    if args.episodes < 1 or not (1 <= args.port <= 65535):
        parser.error("--episodes must be positive and --port must be 1..65535")
    if not HTML_FILE.is_file():
        parser.error(f"Missing web UI file {HTML_FILE}; pull latest repository branch")
    collection = SuiteCollection(args.output, args.episodes, args.seed, args.sidecar, reference)
    # Loopback only; remote users must authenticate through SSH/VPN.
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.collection = collection
    httpd.experiment = collection.datasets.get("libero_10")
    print(f"RECAP web dashboard: http://127.0.0.1:{args.port}", flush=True)
    for item in collection.catalog()["suites"]:
        print(f"{item['name']} Task 0: {item['status']} ({item['episodes']} paired episodes expected)", flush=True)
    print("Private loopback service. From Windows: ssh -N -L "
          f"{args.port}:127.0.0.1:{args.port} muttaqien@10.17.94.130", flush=True)
    try:
        httpd.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        print("Stopping RECAP web dashboard", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
