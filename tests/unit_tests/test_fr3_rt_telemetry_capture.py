"""Offline-only controller telemetry tap tests. No ROS or hardware."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def native_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ needed for standalone C++17 tests")
    root = Path(__file__).resolve().parents[2]
    src = root / "tests/unit_tests/fr3_rt_telemetry_capture_native.cpp"
    binary = tmp_path_factory.mktemp("fr3_telemetry_tap") / "offline_tap"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-pthread", "-I", str(root), str(src), "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=45,
    )
    return binary


@pytest.mark.parametrize("scenario", ["capture", "overflow"])
def test_offline_controller_tap(native_binary, scenario):
    output = subprocess.run(
        [str(native_binary), scenario],
        check=True, capture_output=True, text=True, timeout=30,
    )
    assert f"PASS {scenario}" in output.stdout
