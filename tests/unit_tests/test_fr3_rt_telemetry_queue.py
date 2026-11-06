"""Compile/run offline SPSC design; no ROS modules or real robot commands."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def native_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for optional offline native SPSC tests")
    root = Path(__file__).resolve().parents[2]
    source = root / "tests/unit_tests/fr3_rt_telemetry_queue_native.cpp"
    target = tmp_path_factory.mktemp("fr3_rt_telemetry") / "synthetic_queue_test"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-pthread", "-I", str(root), str(source), "-o", str(target)],
        check=True, capture_output=True, text=True, timeout=45,
    )
    return target


@pytest.mark.parametrize("scenario", ["fifo", "snapshot", "concurrent"])
def test_offline_spsc_scenario(native_binary, scenario):
    result = subprocess.run(
        [str(native_binary), scenario], check=True, capture_output=True,
        text=True, timeout=30,
    )
    assert f"PASS {scenario}" in result.stdout
