#!/usr/bin/env python3
"""Stage an instrumented copy of the NUC FR3 controller, and optionally build it.

Never writes to the live controller workspace, loads plugins, or starts ROS.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shlex
import shutil
import subprocess

SOURCE_DEFAULT = Path.home() / "Desktop/aist-gamma/franka_controller_ws/src/franka_arm_controllers"
DEST_DEFAULT = Path.home() / "fr3_telemetry_isolated_4c2t5"
TARGET_ANCHOR = "  const auto target = *target_buffer_.readFromRT();\n"
TAIL = (
    "  for (size_t i = 0; i < 7; ++i)\n"
    "    command_interfaces_[command_indices_[i]].set_value(torque[i]);\n"
    "  return controller_interface::return_type::OK;\n"
)
CAPTURE = """  // Offline-only telemetry: enqueues post-limit controller reference.
  // Full queue drops telemetry, never commands. No RT disk or ROS calls.
  fr3_rt_telemetry::ArmTickSnapshot record{};
  const auto stamp = [](std::int64_t value) -> std::uint64_t {
    return value > 0 ? static_cast<std::uint64_t>(value) : 0;
  };
  record.controller_tick_seq = telemetry_tick_seq_++;
  record.tick_ns = stamp(now);
  record.activation_ns = stamp(activation);
  record.intent_recv_ns = stamp(intent.received_ns);
  record.target_state_ns = stamp(target.state_ns);
  record.target_command_ns = stamp(target.command_ns);
  record.target_activation_ns = stamp(target.activation_ns);
  // Absent in the original controller, so never fabricate timestamps:
  record.sample_ns = 0;
  record.ik_solve_ns = 0;
  record.intent_axes = intent.axes;
  for (size_t i = 0; i < 7; ++i) {
    record.measured_q_rad[i] = q_[i];
    record.ik_target_q_rad[i] = target.valid ? target.q[i] : 0.0;
    record.reference_before_rad[i] = telemetry_reference_before[i];
    record.reference_after_rad[i] = q_desired_[i];
    record.velocity_limits_rad_s[i] = velocity_limits_[i];
  }
  record.dt_s = std::min(period.seconds(), 0.001);
  record.command_timeout_s = command_timeout_s_;
  const bool intent_fresh = fresh(intent.received_ns);
  const bool state_fresh = fresh(target.state_ns);
  const bool command_fresh = fresh(target.command_ns);
  const bool activation_matches = target.activation_ns == activation;
  record.gate_bits = static_cast<std::uint8_t>(
      (intent.moving ? fr3_rt_telemetry::kMoving : 0) |
      (intent_fresh ? fr3_rt_telemetry::kIntentFresh : 0) |
      (target.valid ? fr3_rt_telemetry::kTargetValid : 0) |
      (activation_matches ? fr3_rt_telemetry::kActivationMatches : 0) |
      (state_fresh ? fr3_rt_telemetry::kStateFresh : 0) |
      (command_fresh ? fr3_rt_telemetry::kCommandFresh : 0));
  record.reference_advanced = intent.moving && intent_fresh && target.valid &&
                              activation_matches && state_fresh && command_fresh;
  (void)telemetry_tap_.capture_after_update(record);
"""


def replace_once(value: str, match: str, replacement: str, name: str) -> str:
    if value.count(match) != 1:
        raise ValueError(f"{name}: unexpected source, found {value.count(match)} anchors")
    return value.replace(match, replacement, 1)


def planned_files(package: Path, helpers: Path) -> tuple[str, str, str, str]:
    header = package / "include/franka_arm_controllers/joint_impedance_ik_controller.hpp"
    source = package / "src/joint_impedance_ik_controller.cpp"
    if not header.is_file() or not source.is_file() or header.is_symlink() or source.is_symlink():
        raise ValueError("Controller header/source missing or symlinked")
    h = header.read_text(encoding="utf-8")
    c = source.read_text(encoding="utf-8")
    for expected in (
        "q_desired_[i] += std::clamp(target.q[i] - q_desired_[i], -step, step);",
        "const auto now = monotonic_ns();",
        "const auto intent = *intent_buffer_.readFromRT();",
    ):
        if expected not in c:
            raise ValueError(f"Unknown controller implementation: missing {expected}")
    h = replace_once(
        h, "#pragma once\n",
        "#pragma once\n#include <franka_arm_controllers/fr3_rt_telemetry_capture.hpp>\n",
        "controller include",
    )
    h = replace_once(
        h, "  realtime_tools::RealtimeBuffer<Target> target_buffer_;",
        "  realtime_tools::RealtimeBuffer<Target> target_buffer_;\n"
        "  fr3_rt_telemetry::ControllerTelemetryTap<1024> telemetry_tap_;\n"
        "  std::uint64_t telemetry_tick_seq_{0};",
        "controller member",
    )
    start = c.find("JointImpedanceIKController::update(")
    stop = c.find("JointImpedanceIKController::on_init()", start)
    if start < 0 or stop <= start:
        raise ValueError("Unable to isolate controller update()")
    prefix, update, suffix = c[:start], c[start:stop], c[stop:]
    update = replace_once(
        update, TARGET_ANCHOR,
        "  const auto telemetry_reference_before = q_desired_;\n" + TARGET_ANCHOR,
        "update target",
    )
    update = replace_once(
        update, TAIL,
        TAIL.replace("  return controller_interface::return_type::OK;\n",
                     CAPTURE + "  return controller_interface::return_type::OK;\n"),
        "update torque tail",
    )
    q = (helpers / "fr3_rt_telemetry_queue.hpp").read_text(encoding="utf-8")
    tap = (helpers / "fr3_rt_telemetry_capture.hpp").read_text(encoding="utf-8")
    tap = tap.replace(
        '#include "toolkits/realworld_check/fr3_rt_telemetry_queue.hpp"',
        '#include <franka_arm_controllers/fr3_rt_telemetry_queue.hpp>',
    )
    if '#include <franka_arm_controllers/fr3_rt_telemetry_queue.hpp>' not in tap:
        raise ValueError("Capture helper include changed")
    return h, prefix + update + suffix, q, tap


def stage(source: Path, destination: Path, helpers: Path) -> Path:
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Missing source package: {source}")
    if destination.is_relative_to(source.parents[1]):
        raise ValueError("Destination cannot be inside the original ROS workspace")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    h, c, q, tap = planned_files(source, helpers)
    cloned = destination / "src/franka_arm_controllers"
    shutil.copytree(source, cloned, symlinks=True)
    include = cloned / "include/franka_arm_controllers"
    (include / "joint_impedance_ik_controller.hpp").write_text(h, encoding="utf-8")
    (cloned / "src/joint_impedance_ik_controller.cpp").write_text(c, encoding="utf-8")
    (include / "fr3_rt_telemetry_queue.hpp").write_text(q, encoding="utf-8")
    (include / "fr3_rt_telemetry_capture.hpp").write_text(tap, encoding="utf-8")
    (destination / "README.txt").write_text(
        "ISOLATED BUILD ONLY. Do not source this install or load its plugin into the live robot.\n"
        "Queue-only snapshots: no consumer, ROS publisher, or dataset export is included.\n",
        encoding="utf-8",
    )
    return destination


def build_isolated(destination: Path, original_ws: Path) -> None:
    overlay = original_ws / "install/setup.bash"
    if not overlay.is_file():
        raise FileNotFoundError(f"Expected ROS overlay not found: {overlay}")
    cmd = (
        "set -eo pipefail; "
        "source /opt/ros/humble/setup.bash; "
        f"source {shlex.quote(str(overlay))}; "
        "colcon "
        f"--log-base {shlex.quote(str(destination / 'log'))} build "
        f"--base-paths {shlex.quote(str(destination / 'src'))} "
        "--packages-select franka_arm_controllers "
        f"--build-base {shlex.quote(str(destination / 'build'))} "
        f"--install-base {shlex.quote(str(destination / 'install'))} "
        "--event-handlers console_direct+"
    )
    print("Building isolated copy only; no live controller changes.", flush=True)
    subprocess.run(["bash", "-c", cmd], cwd=destination, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--out", type=Path, default=DEST_DEFAULT)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    destination = stage(args.source, args.out, Path(__file__).resolve().parent)
    print(f"STAGED: {destination}", flush=True)
    if args.build:
        build_isolated(destination, args.source.expanduser().resolve().parents[1])
        print("ISOLATED BUILD PASSED; live controller untouched.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
