"""One CPU-only test of actual JSONL bytes from the C++ non-RT drain."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_synthetic_recorder_jsonl(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ not installed")
    root = Path(__file__).resolve().parents[2]
    source = root / "toolkits/realworld_check/fr3_telemetry_recorder_demo.cpp"
    binary = tmp_path / "recorder_demo"
    output = tmp_path / "synthetic_controller_telemetry.jsonl"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-I", str(root), str(source), "-o", str(binary)],
        check=True, text=True, capture_output=True, timeout=40,
    )
    completed = subprocess.run(
        [str(binary), str(output)],
        check=True, capture_output=True, text=True, timeout=10,
    )
    assert "saved=2 dropped=1" in completed.stdout
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["schema"] == "fr3_controller_capture_raw_v1" for r in rows)
    assert [r["seq"] for r in rows] == [0, 1]
    assert [r["controller_tick_seq"] for r in rows] == [1, 2]
    assert rows[0]["ik_target_q_rad"] == [0.2] * 7
    assert rows[0]["reference_before_rad"] == [0.0] * 7
    assert rows[0]["reference_after_rad"] == [0.001] * 7
    assert rows[1]["reference_before_rad"] == rows[1]["reference_after_rad"]
    assert rows[0]["ik_solve_ns"] is None
    assert rows[0]["sample_ns"] is None
    assert all(r["training_action"] is None and not r["policy_ready"] for r in rows)
