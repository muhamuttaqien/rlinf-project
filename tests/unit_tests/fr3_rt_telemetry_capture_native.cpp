// Synthetic controller integration; no ROS, hardware, or motor commands.
#include "toolkits/realworld_check/fr3_rt_telemetry_capture.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using fr3_rt_telemetry::ArmTickSnapshot;
using fr3_rt_telemetry::ControllerTelemetryTap;

// Mirrors only the audited q_desired_ rate limit, not actual control.
template <std::size_t Capacity>
void simulated_tick(ControllerTelemetryTap<Capacity>& tap,
                    std::array<double, 7>& q_desired,
                    std::uint64_t tick, bool moving) {
  ArmTickSnapshot sample{};
  sample.controller_tick_seq = tick;
  sample.tick_ns = 1'000'000'000ULL + tick * 1'000'000ULL;
  sample.reference_before_rad = q_desired;
  sample.ik_target_q_rad.fill(0.2);
  sample.velocity_limits_rad_s.fill(1.0);
  sample.intent_axes[0] = moving ? 0.5 : 0.0;
  sample.dt_s = 0.001;
  if (moving) {
    for (std::size_t i = 0; i < 7; ++i) {
      const double step = sample.velocity_limits_rad_s[i] * sample.dt_s;
      q_desired[i] += std::clamp(sample.ik_target_q_rad[i] - q_desired[i], -step, step);
    }
    sample.gate_bits = fr3_rt_telemetry::kMoving |
                       fr3_rt_telemetry::kIntentFresh |
                       fr3_rt_telemetry::kTargetValid |
                       fr3_rt_telemetry::kActivationMatches |
                       fr3_rt_telemetry::kStateFresh |
                       fr3_rt_telemetry::kCommandFresh;
  } else {
    sample.gate_bits = fr3_rt_telemetry::kIntentFresh |
                       fr3_rt_telemetry::kTargetValid |
                       fr3_rt_telemetry::kActivationMatches |
                       fr3_rt_telemetry::kStateFresh |
                       fr3_rt_telemetry::kCommandFresh;
  }
  sample.reference_after_rad = q_desired;
  sample.reference_advanced = moving;
  (void)tap.capture_after_update(sample);
}

void capture_moving_then_hold() {
  ControllerTelemetryTap<4> tap;
  std::array<double, 7> q{};
  simulated_tick(tap, q, 1, true);
  simulated_tick(tap, q, 2, false);
  ArmTickSnapshot first{}, second{};
  assert(tap.drain_one(first) && tap.drain_one(second));
  assert(first.seq == 0 && second.seq == 1);
  assert(first.reference_before_rad[0] == 0.0);
  assert(std::abs(first.reference_after_rad[0] - 0.001) < 1e-12);
  assert(first.reference_advanced);
  assert(second.reference_before_rad == first.reference_after_rad);
  assert(second.reference_after_rad == first.reference_after_rad);
  assert(!second.reference_advanced);
  assert(second.controller_tick_seq == 2);
  assert(tap.dropped_total() == 0);
}

void capture_overflow_and_recovery() {
  ControllerTelemetryTap<2> tap;
  std::array<double, 7> q{};
  simulated_tick(tap, q, 1, true);
  simulated_tick(tap, q, 2, true);
  simulated_tick(tap, q, 3, true);  // full, dropped; does NOT change q update
  assert(std::abs(q[0] - 0.003) < 1e-12);
  assert(tap.dropped_total() == 1);
  ArmTickSnapshot first{}, second{}, fourth{};
  assert(tap.drain_one(first) && tap.drain_one(second));
  assert(first.seq == 0 && second.seq == 1);
  simulated_tick(tap, q, 4, true);
  assert(tap.drain_one(fourth));
  assert(fourth.seq == 3 && fourth.controller_tick_seq == 4);
  assert(fourth.telemetry_drops_total_before_this_try == 1);
  assert(std::abs(fourth.reference_after_rad[0] - 0.004) < 1e-12);
  assert(tap.dropped_total() == 1);
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::string mode = argv[1];
  if (mode == "capture") capture_moving_then_hold();
  else if (mode == "overflow") capture_overflow_and_recovery();
  else return 2;
  std::cout << "PASS " << mode << "\n";
}
