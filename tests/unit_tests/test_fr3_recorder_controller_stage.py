"""Check opt-in non-RT controller instrumentation without ROS hardware."""
import pytest
from toolkits.realworld_check.stage_fr3_recorder_controller import instrument

HEADER = """#pragma once
class JointImpedanceIKController {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
 private:
  void solve_ik_();  // Non-real-time timer callback; never called by update().
  std::uint64_t telemetry_tick_seq_{0};
};
"""
CPP = """#include <franka_arm_controllers/joint_impedance_ik_controller.hpp>
void dummy() { (void)telemetry_tap_.capture_after_update(record); }
CallbackReturn JointImpedanceIKController::on_init() {
  auto_declare<double>("command_timeout", 0.25);
  return CallbackReturn::SUCCESS;
}
CallbackReturn JointImpedanceIKController::on_activate(const rclcpp_lifecycle::State&) {
  active_.store(true);
  return CallbackReturn::SUCCESS;
}
CallbackReturn JointImpedanceIKController::on_deactivate(const rclcpp_lifecycle::State&) {
  active_.store(false);
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}
CallbackReturn JointImpedanceIKController::on_cleanup(const rclcpp_lifecycle::State&) {
  active_.store(false);
  return CallbackReturn::SUCCESS;
}
"""


def test_adds_opt_in_thread_and_shutdown():
    h, c = instrument(HEADER, CPP)
    assert 'auto_declare<std::string>("telemetry_jsonl_path", "");' in c
    assert 'if (!telemetry_path.empty())' in c
    assert "std::thread telemetry_worker_;" in h
    assert "fr3_rt_telemetry::drain_jsonl(telemetry_tap_, stream)" in c
    assert c.count("  stop_recorder_();") == 3  # destructor, deactivation, cleanup
    assert "telemetry_tap_.capture_after_update(record)" in c
    assert "stop_recorder_() noexcept" in c


def test_refuses_uninstrumented_source():
    with pytest.raises(ValueError, match="not the isolated"):
        instrument(HEADER, CPP.replace("telemetry_tap_.capture_after_update(record)",
                                       "nothing()"))
