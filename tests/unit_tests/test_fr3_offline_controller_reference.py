"""Compile a standalone, non-ROS FR3 reference-step model and verify it."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def offline_controller_binary(tmp_path_factory):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("g++ not installed")
    repo = Path(__file__).resolve().parents[2]
    src = repo / "tests/unit_tests/fr3_offline_controller_reference_native.cpp"
    binary = tmp_path_factory.mktemp("fr3_offline_build") / "offline_controller_test"
    subprocess.run(
        [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
         "-I", str(repo), str(src), "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=45,
    )
    return binary


@pytest.mark.parametrize("mode", ["equivalence", "timeout", "overflow"])
def test_offline_controller_reference(offline_controller_binary, mode):
    completed = subprocess.run(
        [str(offline_controller_binary), mode],
        capture_output=True, text=True, check=True, timeout=15,
    )
    assert f"PASS {mode}" in completed.stdout
