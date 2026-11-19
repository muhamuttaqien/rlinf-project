// Offline build verifies the audited reference-gating code path with telemetry.
#include "toolkits/realworld_check/fr3_offline_controller_reference.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using fr3_rt_telemetry::ArmTickSnapshot;
using fr3_rt_telemetry::ControllerTelemetryTap;
using fr3_rt_telemetry::ReferenceStepInput;
using fr3_rt_telemetry::offline_reference_step;

ReferenceStepInput sample() {
  ReferenceStepInput in{};
  in.tick_ns = 1'000'000'000;
  in.activation_ns = 900'000'000;
  in.intent_recv_ns = 999'900'000;
  in.target_state_ns = 999'900'000;
  in.target_command_ns = 999'900'000;
  in.target_activation_ns = 900'000'000;
  in.sample_ns = 999'900'000;
  in.normalized_axes[0] = 0.5;
  in.ik_target_q_rad.fill(0.2);
  in.velocity_limits_rad_s.fill(1.0);
  in.target_valid = true;
  return in;
}

void identical_with_and_without_tap() {
  ControllerTelemetryTap<4> tap;
  std::array<double, 7> reference_a{}, reference_b{};
  auto in = sample();
  for (int i = 0; i < 3; ++i) {
    in.controller_tick_seq = i;
    in.tick_ns += 1'000'000;
    offline_reference_step(reference_a, in, &tap);
    offline_reference_step(reference_b, in,
        static_cast<ControllerTelemetryTap<4>*>(nullptr));
    assert(reference_a == reference_b);
  }
  ArmTickSnapshot r{};
  assert(tap.drain_one(r));
  assert(r.seq == 0 && r.controller_tick_seq == 0);
  assert(r.reference_before_rad[0] == 0);
  assert(std::abs(r.reference_after_rad[0] - 0.001) < 1e-12);
  assert(r.reference_advanced);
  assert(r.ik_solve_ns == 0); // missing in actual controller, not fabricated
}

void timeout_holds_and_records_reason() {
  ControllerTelemetryTap<4> tap;
  std::array<double, 7> ref{};
  auto in = sample();
  in.target_command_ns = 700'000'000; // 300 ms > 250 ms timeout
  offline_reference_step(ref, in, &tap);
  ArmTickSnapshot r{};
  assert(tap.drain_one(r));
  assert(ref[0] == 0.0);
  assert(!r.reference_advanced);
  assert(!(r.gate_bits & fr3_rt_telemetry::kCommandFresh));
}

void loss_does_not_affect_reference() {
  ControllerTelemetryTap<2> tap;
  std::array<double, 7> a{}, b{};
  auto in = sample();
  for (int i = 0; i < 4; ++i) {
    in.controller_tick_seq = i;
    in.tick_ns += 1'000'000;
    offline_reference_step(a, in, &tap);
    offline_reference_step(b, in,
        static_cast<ControllerTelemetryTap<2>*>(nullptr));
  }
  assert(a == b);
  assert(std::abs(a[0] - 0.004) < 1e-12);
  assert(tap.dropped_total() == 2);
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string mode(argv[1]);
  if (mode == "equivalence") identical_with_and_without_tap();
  else if (mode == "timeout") timeout_holds_and_records_reason();
  else if (mode == "overflow") loss_does_not_affect_reference();
  else return 2;
  std::cout << "PASS " << mode << '\n';
}
