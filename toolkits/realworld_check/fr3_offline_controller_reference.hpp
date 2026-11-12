// Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
// Offline reference-step model matching the audited NUC controller logic.
// It is NOT the deployed ROS controller or a physical robot command path.
#pragma once

#include "toolkits/realworld_check/fr3_rt_telemetry_capture.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

namespace fr3_rt_telemetry {

struct ReferenceStepInput {
  std::uint64_t controller_tick_seq{};
  std::uint64_t tick_ns{}, activation_ns{}, intent_recv_ns{};
  std::uint64_t target_state_ns{}, target_command_ns{}, target_activation_ns{};
  std::uint64_t sample_ns{};
  std::array<double, 6> normalized_axes{};
  std::array<double, 7> measured_q_rad{}, ik_target_q_rad{}, velocity_limits_rad_s{};
  double period_s{0.001}, command_timeout_s{0.25};
  bool target_valid{false};
};

// Used only by a standalone, non-ROS test build. The two code paths compute
// the same joint reference; telemetry never determines whether it advances.
// The real controller still uses its own verified update implementation.
template <std::size_t Capacity>
void offline_reference_step(std::array<double, 7>& q_desired,
                            const ReferenceStepInput& in,
                            ControllerTelemetryTap<Capacity>* tap) noexcept {
  const auto fresh = [&](std::uint64_t stamp) noexcept {
    return stamp >= in.activation_ns && in.tick_ns >= stamp &&
           static_cast<double>(in.tick_ns - stamp) * 1e-9 <= in.command_timeout_s;
  };
  const bool moving = std::any_of(
      in.normalized_axes.begin(), in.normalized_axes.end(),
      [](double axis) { return axis != 0.0; });
  const bool intent_fresh = fresh(in.intent_recv_ns);
  const bool activation_matches = in.target_activation_ns == in.activation_ns;
  const bool state_fresh = fresh(in.target_state_ns);
  const bool command_fresh = fresh(in.target_command_ns);
  const bool advance = moving && intent_fresh && in.target_valid &&
                       activation_matches && state_fresh && command_fresh;

  const auto before = q_desired;
  const double dt = std::min(in.period_s, 0.001);
  if (advance) {
    for (std::size_t i = 0; i < 7; ++i) {
      const double step = in.velocity_limits_rad_s[i] * dt;
      q_desired[i] += std::clamp(in.ik_target_q_rad[i] - q_desired[i], -step, step);
    }
  }
  if (tap == nullptr) return;

  ArmTickSnapshot snapshot{};
  snapshot.controller_tick_seq = in.controller_tick_seq;
  snapshot.tick_ns = in.tick_ns;
  snapshot.activation_ns = in.activation_ns;
  snapshot.intent_recv_ns = in.intent_recv_ns;
  snapshot.target_state_ns = in.target_state_ns;
  snapshot.target_command_ns = in.target_command_ns;
  snapshot.target_activation_ns = in.target_activation_ns;
  snapshot.sample_ns = in.sample_ns;
  // The deployed Target struct has no separate IK solve timestamp.
  // Do not invent one; leave ik_solve_ns=0 for unavailable data.
  snapshot.intent_axes = in.normalized_axes;
  snapshot.measured_q_rad = in.measured_q_rad;
  if (in.target_valid) snapshot.ik_target_q_rad = in.ik_target_q_rad;
  snapshot.reference_before_rad = before;
  snapshot.reference_after_rad = q_desired;
  snapshot.velocity_limits_rad_s = in.velocity_limits_rad_s;
  snapshot.dt_s = dt;
  snapshot.command_timeout_s = in.command_timeout_s;
  snapshot.gate_bits = (moving ? kMoving : 0) |
                       (intent_fresh ? kIntentFresh : 0) |
                       (in.target_valid ? kTargetValid : 0) |
                       (activation_matches ? kActivationMatches : 0) |
                       (state_fresh ? kStateFresh : 0) |
                       (command_fresh ? kCommandFresh : 0);
  snapshot.reference_advanced = advance;
  (void)tap->capture_after_update(snapshot);  // Drop if full; never block.
}

}  // namespace fr3_rt_telemetry
