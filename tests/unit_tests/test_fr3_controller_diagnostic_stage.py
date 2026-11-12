"""Controller diagnostic staging is offline; no ROS or motor commands."""

from pathlib import Path

import pytest

from toolkits.realworld_check.fr3_controller_diagnostic_stage import (
    HEADER_REL, SOURCE_REL, instrument, stage,
)


HEADER = """  std::atomic<int64_t> activation_ns_{0};
"""
SOURCE = """  const auto target = *target_buffer_.readFromRT();
  if (intent.moving && fresh(intent.received_ns) && target.valid &&
      target.activation_ns == activation && fresh(target.state_ns) && fresh(target.command_ns)) {
    // A delayed cycle must not turn into a larger reference jump at the 1 kHz boundary.
    const double dt = std::min(period.seconds(), 0.001);
    for (size_t i = 0; i < 7; ++i) {
      const double step = velocity_limits_[i] * dt;
      q_desired_[i] += std::clamp(target.q[i] - q_desired_[i], -step, step);
    }
  }
  for (size_t i = 0; i < 7; ++i)
    command_interfaces_[command_indices_[i]].set_value(torque[i]);
  return controller_interface::return_type::OK;
  ik_timer_ = get_node()->create_wall_timer(std::chrono::duration<double>(1.0 / ik_update_rate_),
                                            [this]() { solve_ik_(); });
  return CallbackReturn::SUCCESS;
  activation_ns_.store(monotonic_ns());
  active_.store(true);
  return CallbackReturn::SUCCESS;
  if (ik_timer_)
    ik_timer_->cancel();
  // Cleanup is non-RT and waits for an in-flight solver before releasing its callback.
  ik_timer_.reset();
  spacemouse_sub_.reset();
  if (std::all_of(intent.axes.begin(), intent.axes.end(),
                  [](double v) { return std::isfinite(v) && std::abs(v) <= 1.0; })) {
  intent_buffer_.writeFromNonRT(intent);
}
  Target target;
  target.state_ns = sample.received_ns;
  target_buffer_.writeFromNonRT(target);
}
"""


def test_staging_preserves_source_and_reports_no_robot_access(tmp_path):
    workspace = tmp_path / "workspace"
    output = tmp_path / "review"
    (workspace / HEADER_REL).parent.mkdir(parents=True)
    (workspace / SOURCE_REL).parent.mkdir(parents=True)
    (workspace / HEADER_REL).write_text(HEADER)
    (workspace / SOURCE_REL).write_text(SOURCE)

    result = stage(workspace, output)
    assert result["source_workspace_unchanged"] is True
    assert result["running_controller_unchanged"] is True
    assert result["robot_commands_sent"] == 0
    assert (workspace / HEADER_REL).read_text() == HEADER
    assert (workspace / SOURCE_REL).read_text() == SOURCE
    patch = (output / "controller_diagnostics_only.patch").read_text()
    assert "diag_rx_" in patch
    assert "diag_ik_valid_" in patch
    assert "diag_ref_advanced_ticks_" in patch
    assert "diag_effort_write_ticks_" in patch
    assert "FR3_DIAG_V1" in patch
    import json
    assert json.loads((output / "stage_report.json").read_text()) == result
    with pytest.raises(FileExistsError):
        stage(workspace, output)


def test_unexpected_source_revision_fails_closed():
    with pytest.raises(ValueError, match="Unexpected controller revision"):
        instrument(HEADER, SOURCE.replace("target_buffer_.readFromRT()",
                                          "different_target_buffer()"))


def test_duplicate_instrumentation_refused():
    amended_header, amended_source = instrument(HEADER, SOURCE)
    assert "RCLCPP_INFO" in amended_source
    with pytest.raises(ValueError, match="already instrumented"):
        instrument(amended_header, amended_source)
