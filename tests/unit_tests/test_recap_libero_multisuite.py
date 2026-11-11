"""Smoke tests for five-suite read-only RECAP browser dataset selection.

No simulator, GPU, model, video decoding, or network connection is required.
"""

import csv
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from toolkits.recap_libero_web_dashboard import (
    Handler, SuiteCollection, ThreadingHTTPServer,
)
from toolkits.recap_libero_multisuite_prepare import output_for_suite


def create_paired_run(folder: Path, suite: str, successes=(0, 1), seed=7):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "baseline").mkdir(exist_ok=True)
    (folder / "recap").mkdir(exist_ok=True)
    (folder / "run_config.json").write_text(json.dumps({
        "seed": seed, "episodes": 2, "suite": suite, "task_index": 0,
        "value_observations": True, "task_description": "pick up the target object",
    }), encoding="utf-8")
    rows = []
    for ep in (1, 2):
        row = {"episode": ep, "seed": seed + ep - 1}
        for policy in ("baseline", "recap"):
            good = (successes[ep - 1] if policy == "recap" else 0)
            relative = f"{policy}/task0_episode{ep}_{'success' if good else 'failure'}.mp4"
            (folder / relative).write_bytes(b"\x00\x00\x00\x18ftypisomFAKE-VIDEO-BYTES")
            row[f"{policy}_video"] = relative
            row[f"{policy}_success"] = good
        rows.append(row)
    with (folder / "paired_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MultiSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "recap_sim7_value_comparison_20episodes"
        create_paired_run(self.root, "libero_10")
        create_paired_run(output_for_suite(self.root, "libero_spatial"), "libero_spatial")
        self.collection = SuiteCollection(self.root, 2, 7, Path(self.temp.name) / "missing.parquet", (7, 19, 30))

    def test_five_categories_and_independent_recordings(self):
        catalog = self.collection.catalog()["suites"]
        self.assertEqual(len(catalog), 5)
        self.assertEqual(sum(bool(x["ready"]) for x in catalog), 2)
        self.assertEqual(self.collection.summary("libero_spatial")["category"], "LIBERO-Spatial")
        self.assertTrue(self.collection.summary("libero_10")["available"])
        self.assertFalse(self.collection.summary("libero_90")["available"])
        self.assertEqual(self.collection.summary("libero_90")["rows"], [])
        self.assertEqual(self.collection.summary("libero_10")["reference"]["trials"], 30)
        self.assertIsNone(self.collection.summary("libero_spatial")["reference"])

    def test_no_cross_suite_video_reuse(self):
        spatial = self.collection.episode("libero_spatial", 1)
        long = self.collection.episode("libero_10", 1)
        self.assertEqual(spatial["baseline"]["url"], "/media/libero_spatial/baseline/1")
        self.assertEqual(long["baseline"]["url"], "/media/libero_10/baseline/1")
        self.assertFalse(self.collection.training("libero_spatial", None)["available"])
        with self.assertRaisesRegex(ValueError, "no recorded"):
            self.collection.episode("libero_goal", 1)
        with self.assertRaisesRegex(ValueError, "Unknown LIBERO suite"):
            self.collection.summary("../private")

    def test_manifest_identity_enforced(self):
        mismatched = output_for_suite(self.root, "libero_object")
        create_paired_run(mismatched, "libero_spatial")
        c = SuiteCollection(self.root, 2, 7, Path(self.temp.name) / "missing.parquet", None)
        self.assertFalse(c.summary("libero_object")["available"])
        status = next(x for x in c.catalog()["suites"] if x["id"] == "libero_object")
        self.assertEqual(status["status"], "Invalid data")

    def test_goal_specialist_override_is_isolated(self):
        specialist = Path(self.temp.name) / "goal_specialist"
        create_paired_run(specialist, "libero_goal")
        c = SuiteCollection(
            self.root, 2, 7, Path(self.temp.name) / "missing.parquet", None,
            goal_specialist_output=specialist,
        )
        self.assertTrue(c.summary("libero_goal")["available"])
        self.assertEqual(c.summary("libero_goal")["policy_variant"], "Goal Task 0 specialist")
        self.assertIn("Goal-trained RECAP", next(
            x["focus"] for x in c.catalog()["suites"] if x["id"] == "libero_goal"
        ))
        self.assertEqual(c.episode("libero_goal", 1)["baseline"]["url"], "/media/libero_goal/baseline/1")
        self.assertTrue(c.summary("libero_10")["available"])

    def test_comparison_color_tracks_success_rate(self):
        html = (
            Path(__file__).resolve().parents[2]
            / "toolkits/recap_libero_web_dashboard.html"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "const recapUnderperforms=s.recap_successes<s.baseline_successes;",
            html,
        )
        self.assertIn(
            "$('recapRate').classList.toggle('negative',recapUnderperforms);",
            html,
        )
        self.assertIn(
            "$('deltaRate').classList.toggle('negative',delta<0);",
            html,
        )
        self.assertIn(
            ".metric.recap .metric-number.negative,.metric.delta .metric-number.negative{color:var(--red)}",
            html,
        )

    def test_http_suite_routes_and_ranges(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.collection = self.collection
        server.experiment = self.collection.datasets["libero_10"]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(2)))
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/suites") as stream:
            self.assertEqual(len(json.load(stream)["suites"]), 5)
        with urlopen(base + "/api/summary?suite=libero_goal") as stream:
            self.assertFalse(json.load(stream)["available"])
        with urlopen(base + "/api/episode?suite=libero_spatial&episode=1") as stream:
            ep = json.load(stream)
            self.assertIn("/libero_spatial/", ep["recap"]["url"])
        with urlopen(Request(base + "/media/libero_spatial/recap/1",
                             headers={"Range": "bytes=3-8"})) as stream:
            self.assertEqual(stream.status, 206)
            self.assertEqual(stream.read(), b"\x18ftypi")
        with self.assertRaises(HTTPError) as error:
            urlopen(base + "/media/libero_goal/baseline/1")
        self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
