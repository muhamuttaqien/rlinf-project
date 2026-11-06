"""Offline staging must not edit original controller files or guess changed source."""
import pytest
from toolkits.realworld_check import stage_fr3_offline_telemetry as stager

HEADER = "#pragma once\n  realtime_tools::RealtimeBuffer<Target> target_buffer_;\n"
CPP = """controller_interface::return_type JointImpedanceIKController::update(
    const rclcpp::Time&, const rclcpp::Duration& period) {
  const auto now = monotonic_ns();
  const auto intent = *intent_buffer_.readFromRT();
  const auto target = *target_buffer_.readFromRT();
  q_desired_[i] += std::clamp(target.q[i] - q_desired_[i], -step, step);
  for (size_t i = 0; i < 7; ++i)
    command_interfaces_[command_indices_[i]].set_value(torque[i]);
  return controller_interface::return_type::OK;
}
CallbackReturn JointImpedanceIKController::on_init() { return CallbackReturn::SUCCESS; }
"""


def sources(tmp_path):
    src = tmp_path / "real_ws/src/franka_arm_controllers"
    (src / "include/franka_arm_controllers").mkdir(parents=True)
    (src / "src").mkdir()
    (src / "include/franka_arm_controllers/joint_impedance_ik_controller.hpp").write_text(HEADER)
    (src / "src/joint_impedance_ik_controller.cpp").write_text(CPP)
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    (helpers / "fr3_rt_telemetry_queue.hpp").write_text("// queue\n")
    (helpers / "fr3_rt_telemetry_capture.hpp").write_text(
        '#include "toolkits/realworld_check/fr3_rt_telemetry_queue.hpp"\n')
    return src, helpers


def test_copy_only(tmp_path):
    src, helpers = sources(tmp_path)
    dest = tmp_path / "isolated"
    stager.stage(src, dest, helpers)
    edited = (dest / "src/franka_arm_controllers/src/joint_impedance_ik_controller.cpp").read_text()
    assert "telemetry_tap_.capture_after_update(record)" in edited
    assert "const auto telemetry_reference_before = q_desired_;" in edited
    assert "record.ik_solve_ns = 0;" in edited
    assert (src / "src/joint_impedance_ik_controller.cpp").read_text() == CPP
    with pytest.raises(FileExistsError):
        stager.stage(src, dest, helpers)


def test_unknown_source_fails_closed(tmp_path):
    src, helpers = sources(tmp_path)
    (src / "src/joint_impedance_ik_controller.cpp").write_text("int main() {}\n")
    dest = tmp_path / "isolated"
    with pytest.raises(ValueError, match="Unknown controller"):
        stager.stage(src, dest, helpers)
    assert not dest.exists()
