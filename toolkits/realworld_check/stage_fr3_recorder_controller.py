#!/usr/bin/env python3
"""Add a non-RT JSONL consumer to the already isolated FR3 telemetry controller.

Clones a new isolated package. Never modifies or loads the running controller.
Recording is DISABLED by default (empty telemetry_jsonl_path parameter).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil
import subprocess

BASE = Path.home() / "fr3_telemetry_isolated_4c2t5/src/franka_arm_controllers"
OUTPUT = Path.home() / "fr3_telemetry_isolated_4c2t7"
ORIGINAL_OVERLAY = Path.home() / "Desktop/aist-gamma/franka_controller_ws/install/setup.bash"

START_WORKER = """  // Non-real-time consumer is opt-in; empty path is the safe default.
  const auto telemetry_path = get_node()->get_parameter("telemetry_jsonl_path").as_string();
  if (!telemetry_path.empty()) {
    telemetry_worker_running_.store(true);
    telemetry_worker_ = std::thread([this, telemetry_path]() {
      try {
        std::ofstream stream(telemetry_path, std::ios::out | std::ios::trunc);
        if (!stream) {
          RCLCPP_ERROR(get_node()->get_logger(), "Cannot open telemetry JSONL output");
          telemetry_worker_running_.store(false);
          return;
        }
        while (telemetry_worker_running_.load()) {
          (void)fr3_rt_telemetry::drain_jsonl(telemetry_tap_, stream);
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        (void)fr3_rt_telemetry::drain_jsonl(telemetry_tap_, stream);
        stream.flush();
        RCLCPP_INFO(get_node()->get_logger(),
                    "Telemetry finished, dropped=%llu",
                    static_cast<unsigned long long>(telemetry_tap_.dropped_total()));
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_node()->get_logger(), "Non-RT telemetry recorder: %s", e.what());
      }
      telemetry_worker_running_.store(false);
    });
  }
"""

STOP_AND_DTOR = """void JointImpedanceIKController::stop_recorder_() noexcept {
  telemetry_worker_running_.store(false);
  if (telemetry_worker_.joinable()) telemetry_worker_.join();
}

JointImpedanceIKController::~JointImpedanceIKController() {
  stop_recorder_();
}

"""


def once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, got {count}; aborting")
    return source.replace(anchor, replacement, 1)


def instrument(header: str, cpp: str) -> tuple[str, str]:
    # Fail closed if fed a source that was not instrumented in 4C.2T.5.
    if "telemetry_tap_.capture_after_update(record)" not in cpp:
        raise ValueError("This source is not the isolated 4C.2T.5 telemetry controller")
    header = once(
        header, "#pragma once\n",
        "#pragma once\n#include <thread>\n",
        "header include",
    )
    header = once(
        header, "  using Vector7d = Eigen::Matrix<double, 7, 1>;",
        "  using Vector7d = Eigen::Matrix<double, 7, 1>;\n"
        "  ~JointImpedanceIKController() override;",
        "destructor",
    )
    header = once(
        header,
        "  std::uint64_t telemetry_tick_seq_{0};",
        "  std::uint64_t telemetry_tick_seq_{0};\n"
        "  std::atomic<bool> telemetry_worker_running_{false};\n"
        "  std::thread telemetry_worker_;",
        "thread members",
    )
    header = once(
        header,
        "  void solve_ik_();  // Non-real-time timer callback; never called by update().",
        "  void solve_ik_();  // Non-real-time timer callback; never called by update().\n"
        "  void stop_recorder_() noexcept;",
        "stop method",
    )
    cpp = once(
        cpp, "#include <franka_arm_controllers/joint_impedance_ik_controller.hpp>",
        "#include <franka_arm_controllers/joint_impedance_ik_controller.hpp>\n"
        "#include <franka_arm_controllers/fr3_rt_telemetry_recorder.hpp>\n"
        "#include <exception>\n#include <fstream>\n#include <thread>",
        "cpp includes",
    )
    cpp = once(
        cpp, "  auto_declare<double>(\"command_timeout\", 0.25);",
        "  auto_declare<double>(\"command_timeout\", 0.25);\n"
        "  auto_declare<std::string>(\"telemetry_jsonl_path\", \"\");",
        "recorder parameter default disabled",
    )
    cpp = once(
        cpp, "  active_.store(true);\n  return CallbackReturn::SUCCESS;",
        "  active_.store(true);\n" + START_WORKER + "  return CallbackReturn::SUCCESS;",
        "activation",
    )
    cpp = once(
        cpp,
        "CallbackReturn JointImpedanceIKController::on_deactivate(const rclcpp_lifecycle::State&) {\n"
        "  active_.store(false);",
        "CallbackReturn JointImpedanceIKController::on_deactivate(const rclcpp_lifecycle::State&) {\n"
        "  active_.store(false);\n  stop_recorder_();",
        "deactivation",
    )
    cpp = once(
        cpp,
        "CallbackReturn JointImpedanceIKController::on_cleanup(const rclcpp_lifecycle::State&) {\n"
        "  active_.store(false);",
        "CallbackReturn JointImpedanceIKController::on_cleanup(const rclcpp_lifecycle::State&) {\n"
        "  active_.store(false);\n  stop_recorder_();",
        "cleanup",
    )
    # Place both methods at namespace scope, before the existing on_init().
    init = "CallbackReturn JointImpedanceIKController::on_init()"
    cpp = once(cpp, init, STOP_AND_DTOR + init, "recorder methods")
    return header, cpp


def stage(source: Path, destination: Path, recorder_header: Path) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Missing staged controller: {source}")
    if destination == source or destination.is_relative_to(source.parents[1]):
        raise ValueError("Output must not be inside the previous isolated workspace")
    if destination.exists():
        raise FileExistsError(f"Will not overwrite existing directory: {destination}")
    hpath = source / "include/franka_arm_controllers/joint_impedance_ik_controller.hpp"
    cpath = source / "src/joint_impedance_ik_controller.cpp"
    h, cpp = instrument(hpath.read_text(), cpath.read_text())
    rec = recorder_header.read_text()
    old = '#include "toolkits/realworld_check/fr3_rt_telemetry_capture.hpp"'
    if rec.count(old) != 1:
        raise ValueError("Unexpected non-RT recorder header include")
    rec = rec.replace(old, '#include <franka_arm_controllers/fr3_rt_telemetry_capture.hpp>')
    target = destination / "src/franka_arm_controllers"
    shutil.copytree(source, target, symlinks=True)
    include = target / "include/franka_arm_controllers"
    (include / "joint_impedance_ik_controller.hpp").write_text(h)
    (target / "src/joint_impedance_ik_controller.cpp").write_text(cpp)
    (include / "fr3_rt_telemetry_recorder.hpp").write_text(rec)
    (destination / "README.txt").write_text(
        "Build only; never source, load, or deploy this controller in a live robot.\n"
        "Non-RT recorder defaults OFF unless telemetry_jsonl_path explicitly set.\n"
        "No physical/RT safety qualification has been performed.\n")
    return destination


def build(destination: Path) -> None:
    if not ORIGINAL_OVERLAY.is_file():
        raise FileNotFoundError(f"Missing controller dependency overlay: {ORIGINAL_OVERLAY}")
    d = shlex.quote(str(destination))
    cmd = (
        "set -eo pipefail; source /opt/ros/humble/setup.bash; "
        f"source {shlex.quote(str(ORIGINAL_OVERLAY))}; "
        f"colcon --log-base {d}/log build --base-paths {d}/src "
        "--packages-select franka_arm_controllers "
        f"--build-base {d}/build --install-base {d}/install "
        "--cmake-args -DBUILD_TESTING=OFF --event-handlers console_direct+"
    )
    subprocess.run(["bash", "-c", cmd], cwd=destination, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--recorder-header", type=Path,
                        default=Path(__file__).resolve().parent / "fr3_rt_telemetry_recorder.hpp")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    destination = stage(args.source, args.out, args.recorder_header)
    print(f"STAGED: {destination}", flush=True)
    if args.build:
        build(destination)
        print("ISOLATED RECORDER CONTROLLER BUILD PASSED; live FR3 untouched.", flush=True)


if __name__ == "__main__":
    main()
