# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Stage, but NEVER install, a read-only FR3 controller diagnostic source diff.

This generator reads the NUC controller source and writes two instrumented
C++ files plus unified diffs under an output directory. The source workspace,
running process, ROS graph, and motor commands are NOT modified.

Counters are relaxed atomics in the input callback, IK timer and RT update.
Logging is ONLY from a separate 1 Hz non-RT wall timer. The implementation
does not alter any command calculation or safety gate, but a new controller
build still requires independent review and an approved maintenance window.

The exact anchors are intentionally fail-closed for unexpected NUC revisions.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

HEADER_REL = Path(
    "src/franka_arm_controllers/include/franka_arm_controllers/"
    "joint_impedance_ik_controller.hpp"
)
SOURCE_REL = Path(
    "src/franka_arm_controllers/src/joint_impedance_ik_controller.cpp"
)
MARKER = "FR3_DIAG_V1"


def replace_one(content: str, old: str, new: str, name: str) -> str:
    count = content.count(old)
    if count != 1:
        raise ValueError(f"Unexpected controller revision at {name}: matched {count} anchors")
    return content.replace(old, new, 1)


def instrument(header: str, source: str) -> tuple[str, str]:
    if MARKER in header or MARKER in source:
        raise ValueError("Controller appears already instrumented: refusing duplicate patch")

    header = replace_one(
        header,
        "  std::atomic<int64_t> activation_ns_{0};",
        """  std::atomic<int64_t> activation_ns_{0};
  // FR3_DIAG_V1: observability only; NO motor command path or gain changes.
  std::atomic<uint64_t> diag_rx_{0};
  std::atomic<uint64_t> diag_moving_rx_{0};
  std::atomic<uint64_t> diag_invalid_rx_{0};
  std::atomic<uint64_t> diag_ik_attempt_{0};
  std::atomic<uint64_t> diag_ik_valid_{0};
  std::atomic<uint64_t> diag_ik_failed_{0};
  std::atomic<uint64_t> diag_gate_ticks_{0};
  std::atomic<uint64_t> diag_ref_advanced_ticks_{0};
  std::atomic<uint64_t> diag_effort_write_ticks_{0};
  std::atomic<double> diag_last_input_x_{0.0};
  std::atomic<double> diag_last_ref_step_rad_{0.0};
  rclcpp::TimerBase::SharedPtr diag_timer_;""",
        "diagnostic atomic members",
    )

    source = replace_one(
        source,
        """  const auto target = *target_buffer_.readFromRT();
  if (intent.moving && fresh(intent.received_ns) && target.valid &&
      target.activation_ns == activation && fresh(target.state_ns) && fresh(target.command_ns)) {
    // A delayed cycle must not turn into a larger reference jump at the 1 kHz boundary.
    const double dt = std::min(period.seconds(), 0.001);
    for (size_t i = 0; i < 7; ++i) {
      const double step = velocity_limits_[i] * dt;
      q_desired_[i] += std::clamp(target.q[i] - q_desired_[i], -step, step);
    }
  }""",
        """  const auto target = *target_buffer_.readFromRT();
  if (intent.moving && fresh(intent.received_ns) && target.valid &&
      target.activation_ns == activation && fresh(target.state_ns) && fresh(target.command_ns)) {
    // A delayed cycle must not turn into a larger reference jump at the 1 kHz boundary.
    const double dt = std::min(period.seconds(), 0.001);
    double largest_reference_step = 0.0;  // FR3_DIAG_V1 only.
    for (size_t i = 0; i < 7; ++i) {
      const double step = velocity_limits_[i] * dt;
      const double before = q_desired_[i];
      q_desired_[i] += std::clamp(target.q[i] - q_desired_[i], -step, step);
      largest_reference_step =
          std::max(largest_reference_step, std::abs(q_desired_[i] - before));
    }
    diag_gate_ticks_.fetch_add(1, std::memory_order_relaxed);
    diag_last_ref_step_rad_.store(largest_reference_step, std::memory_order_relaxed);
    if (largest_reference_step > 1e-12)
      diag_ref_advanced_ticks_.fetch_add(1, std::memory_order_relaxed);
  }""",
        "RT target gate and reference step",
    )

    source = replace_one(
        source,
        """  for (size_t i = 0; i < 7; ++i)
    command_interfaces_[command_indices_[i]].set_value(torque[i]);
  return controller_interface::return_type::OK;""",
        """  for (size_t i = 0; i < 7; ++i)
    command_interfaces_[command_indices_[i]].set_value(torque[i]);
  // FR3_DIAG_V1: counts set_value() calls, NOT hardware torque acceptance.
  diag_effort_write_ticks_.fetch_add(1, std::memory_order_relaxed);
  return controller_interface::return_type::OK;""",
        "RT effort writes",
    )

    source = replace_one(
        source,
        """  ik_timer_ = get_node()->create_wall_timer(std::chrono::duration<double>(1.0 / ik_update_rate_),
                                            [this]() { solve_ik_(); });
  return CallbackReturn::SUCCESS;""",
        """  ik_timer_ = get_node()->create_wall_timer(std::chrono::duration<double>(1.0 / ik_update_rate_),
                                            [this]() { solve_ik_(); });
  // FR3_DIAG_V1: log once a second outside the RT update callback.
  diag_timer_ = get_node()->create_wall_timer(std::chrono::seconds(1), [this]() {
    if (!active_.load(std::memory_order_relaxed))
      return;
    RCLCPP_INFO(
        get_node()->get_logger(),
        "FR3_DIAG_V1 rx=%llu moving_rx=%llu invalid_rx=%llu last_x=%.6f "
        "ik_attempt=%llu ik_valid=%llu ik_failed=%llu "
        "gate_ticks=%llu ref_advanced_ticks=%llu effort_write_ticks=%llu "
        "last_reference_step_rad=%.9f",
        static_cast<unsigned long long>(diag_rx_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(diag_moving_rx_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(diag_invalid_rx_.load(std::memory_order_relaxed)),
        diag_last_input_x_.load(std::memory_order_relaxed),
        static_cast<unsigned long long>(diag_ik_attempt_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(diag_ik_valid_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(diag_ik_failed_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(diag_gate_ticks_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(diag_ref_advanced_ticks_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(diag_effort_write_ticks_.load(std::memory_order_relaxed)),
        diag_last_ref_step_rad_.load(std::memory_order_relaxed));
  });
  return CallbackReturn::SUCCESS;""",
        "non-RT reporting timer",
    )

    source = replace_one(
        source,
        """  activation_ns_.store(monotonic_ns());
  active_.store(true);
  return CallbackReturn::SUCCESS;""",
        """  // FR3_DIAG_V1: reset cumulative counters at each controller activation.
  diag_rx_.store(0, std::memory_order_relaxed);
  diag_moving_rx_.store(0, std::memory_order_relaxed);
  diag_invalid_rx_.store(0, std::memory_order_relaxed);
  diag_ik_attempt_.store(0, std::memory_order_relaxed);
  diag_ik_valid_.store(0, std::memory_order_relaxed);
  diag_ik_failed_.store(0, std::memory_order_relaxed);
  diag_gate_ticks_.store(0, std::memory_order_relaxed);
  diag_ref_advanced_ticks_.store(0, std::memory_order_relaxed);
  diag_effort_write_ticks_.store(0, std::memory_order_relaxed);
  diag_last_input_x_.store(0.0, std::memory_order_relaxed);
  diag_last_ref_step_rad_.store(0.0, std::memory_order_relaxed);
  activation_ns_.store(monotonic_ns());
  active_.store(true);
  return CallbackReturn::SUCCESS;""",
        "activation and counter reset",
    )

    source = replace_one(
        source,
        """  if (ik_timer_)
    ik_timer_->cancel();
  // Cleanup is non-RT and waits for an in-flight solver before releasing its callback.""",
        """  if (ik_timer_)
    ik_timer_->cancel();
  if (diag_timer_)
    diag_timer_->cancel();
  // Cleanup is non-RT and waits for an in-flight solver before releasing its callback.""",
        "diagnostic timer cleanup cancellation",
    )

    source = replace_one(
        source,
        """  ik_timer_.reset();
  spacemouse_sub_.reset();""",
        """  ik_timer_.reset();
  diag_timer_.reset();
  spacemouse_sub_.reset();""",
        "diagnostic timer cleanup reset",
    )

    source = replace_one(
        source,
        """  intent_buffer_.writeFromNonRT(intent);
}""",
        """  // FR3_DIAG_V1: callback receipt is not controller acceptance or motion.
  diag_rx_.fetch_add(1, std::memory_order_relaxed);
  if (intent.moving) {
    diag_moving_rx_.fetch_add(1, std::memory_order_relaxed);
    diag_last_input_x_.store(intent.axes[0], std::memory_order_relaxed);
  } else if (std::any_of(
                 intent.axes.begin(), intent.axes.end(),
                 [](double value) { return !std::isfinite(value); })) {
    diag_invalid_rx_.fetch_add(1, std::memory_order_relaxed);
  }
  intent_buffer_.writeFromNonRT(intent);
}""",
        "SpaceMouse subscription input receipt",
    )
    # The existing callback zeroes rejected axes. Track original validity by
    # using the original Twist message, not the sanitized Intent copy.
    source = replace_one(
        source,
        """  } else if (std::any_of(
                 intent.axes.begin(), intent.axes.end(),
                 [](double value) { return !std::isfinite(value); })) {
    diag_invalid_rx_.fetch_add(1, std::memory_order_relaxed);
  }""",
        """  } else if (!std::all_of(
                 std::array<double, 6>{msg->linear.x, msg->linear.y, msg->linear.z,
                                       msg->angular.x, msg->angular.y, msg->angular.z}.begin(),
                 std::array<double, 6>{msg->linear.x, msg->linear.y, msg->linear.z,
                                       msg->angular.x, msg->angular.y, msg->angular.z}.end(),
                 [](double value) { return std::isfinite(value) && std::abs(value) <= 1.0; })) {
    diag_invalid_rx_.fetch_add(1, std::memory_order_relaxed);
  }""",
        "invalid request counter",
    )
    # Do not iterate begin/end of distinct temporaries; use the source
    # controller's already validated branch through a new local boolean.
    source = replace_one(
        source,
        """  if (std::all_of(intent.axes.begin(), intent.axes.end(),
                  [](double v) { return std::isfinite(v) && std::abs(v) <= 1.0; })) {""",
        """  const bool accepted_range = std::all_of(intent.axes.begin(), intent.axes.end(),
      [](double v) { return std::isfinite(v) && std::abs(v) <= 1.0; });
  if (accepted_range) {""",
        "accepted normalized callback bounds",
    )
    source = replace_one(
        source,
        """  } else if (!std::all_of(
                 std::array<double, 6>{msg->linear.x, msg->linear.y, msg->linear.z,
                                       msg->angular.x, msg->angular.y, msg->angular.z}.begin(),
                 std::array<double, 6>{msg->linear.x, msg->linear.y, msg->linear.z,
                                       msg->angular.x, msg->angular.y, msg->angular.z}.end(),
                 [](double value) { return std::isfinite(value) && std::abs(value) <= 1.0; })) {""",
        """  } else if (!accepted_range) {""",
        "use original range validation for invalid count",
    )

    source = replace_one(
        source,
        """  Target target;
  target.state_ns = sample.received_ns;""",
        """  // FR3_DIAG_V1: counts only IK computations reaching target construction.
  diag_ik_attempt_.fetch_add(1, std::memory_order_relaxed);
  Target target;
  target.state_ns = sample.received_ns;""",
        "IK solver attempt",
    )

    source = replace_one(
        source,
        """  target_buffer_.writeFromNonRT(target);
}""",
        """  // FR3_DIAG_V1: IK valid means a finite joint target within joint limits.
  if (target.valid)
    diag_ik_valid_.fetch_add(1, std::memory_order_relaxed);
  else
    diag_ik_failed_.fetch_add(1, std::memory_order_relaxed);
  target_buffer_.writeFromNonRT(target);
}""",
        "IK target validity",
    )
    return header, source


def unified(name: Path, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile="a/" + str(name),
        tofile="b/" + str(name),
    ))


def stage(workspace: Path, output: Path) -> dict:
    root = workspace.expanduser().resolve()
    dest = output.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Controller workspace not found: {root}")
    if dest.exists():
        raise FileExistsError("Output already exists; refuse overwrite of staged evidence")
    hpath, cpath = root / HEADER_REL, root / SOURCE_REL
    hbefore = hpath.read_text(encoding="utf-8")
    cbefore = cpath.read_text(encoding="utf-8")
    hafter, cafter = instrument(hbefore, cbefore)
    # Only create outputs after all exact source anchors were validated.
    dest.mkdir(parents=True)
    for rel, text in ((HEADER_REL, hafter), (SOURCE_REL, cafter)):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    diff = unified(HEADER_REL, hbefore, hafter) + unified(SOURCE_REL, cbefore, cafter)
    (dest / "controller_diagnostics_only.patch").write_text(diff, encoding="utf-8")
    result = {
        "mode": "offline_source_staging_only",
        "source_workspace_unchanged": True,
        "running_controller_unchanged": True,
        "robot_commands_sent": 0,
        "output_directory": str(dest),
        "patched_header": str(dest / HEADER_REL),
        "patched_source": str(dest / SOURCE_REL),
        "review_patch": str(dest / "controller_diagnostics_only.patch"),
        "manual_build_and_live_approval_required": True,
    }
    (dest / "stage_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.workspace, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
